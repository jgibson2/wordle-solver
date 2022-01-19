import copy
import functools
import itertools
import sys

import click
import json
import numpy as np
import collections
import enum
import re

import concurrent.futures


class GuessResult(enum.Enum):
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2

    @staticmethod
    def format_result(guess):
        replacements = {
            GuessResult.ABSENT: '✘',
            GuessResult.PRESENT: '□',
            GuessResult.CORRECT: '✔',
        }
        return ''.join((replacements[r] for r in guess))


class Outcome(enum.Enum):
    WIN = 1
    LOSE = 2


class Wordle:
    def __init__(self, solution, valid_solutions, valid_guesses, max_attempts=6):
        self.solution = solution
        if solution not in valid_solutions:
            raise ValueError("solution not in valid solutions")
        self.valid_guesses = set(valid_guesses)
        self.valid_solutions = set(valid_solutions)
        self.max_attempts = max_attempts
        self.num_attempts = 0

    def guess(self, guess):
        if self.num_attempts > self.max_attempts:
            raise ValueError("guesses exceeds max guesses")
        if len(guess) != len(self.solution):
            raise ValueError("length of guess and solution do not match")
        if guess not in self.valid_guesses and guess not in self.valid_solutions:
            raise ValueError("guess not in valid guesses")
        outcome = None
        result = Wordle.check(guess, self.solution)
        self.num_attempts += 1
        if all([r == GuessResult.CORRECT for r in result]):
            outcome = Outcome.WIN
        elif self.num_attempts == self.max_attempts:
            outcome = Outcome.LOSE
        return result, outcome

    def num_letters(self):
        return len(self.solution)

    @staticmethod
    def check(guess, solution):
        result = [GuessResult.ABSENT] * len(guess)
        current_counts = collections.Counter(list(solution))
        for i, (guess_letter, solution_letter) in enumerate(zip(list(guess), solution)):
            if guess_letter == solution_letter:
                result[i] = GuessResult.CORRECT
                current_counts[guess_letter] -= 1
        for i, (guess_letter, solution_letter) in enumerate(zip(list(guess), solution)):
            if result[i] == GuessResult.ABSENT and guess_letter in current_counts and current_counts[guess_letter] > 0:
                result[i] = GuessResult.PRESENT
                current_counts[guess_letter] -= 1
        return result


class WordleSolver:
    ALPHABET = tuple((chr(ord('a') + i) for i in range(26)))

    def __init__(self, num_letters, valid_solutions, valid_guesses, cached_first_guess=None, num_guesses=1000, hard_mode=False):
        self.num_letters = num_letters
        self.valid_solutions = set(valid_solutions)
        self.valid_guesses = set(valid_guesses)
        self.letter_sets = [set(list(WordleSolver.ALPHABET)) for _ in range(self.num_letters)]
        self.present_character_counts = {}
        self.known = [None] * self.num_letters
        self.past_guesses = []
        self.cached_guess = cached_first_guess
        self.num_guesses = num_guesses
        self.hard_mode = hard_mode

    def update(self, guess, guess_result):
        guess = guess.lower()
        if len(guess) != self.num_letters:
            raise ValueError("Guess length does not match number of letters")
        self.past_guesses.append(guess)

        self._update_in_place(guess, guess_result, self.known, self.letter_sets, self.present_character_counts)

        for soln in self._find_invalidated_solutions(
                self.valid_solutions,
                self.known,
                self.letter_sets,
                self.present_character_counts,
        ):
            self.valid_solutions.remove(soln)

        if self.hard_mode:
            for guess in self._find_invalidated_solutions(
                    self.valid_guesses,
                    self.known,
                    self.letter_sets,
                    self.present_character_counts,
            ):
                self.valid_guesses.remove(guess)

    def _update_in_place(self, guess, guess_result, known, letter_sets, present_character_counts):
        # update present counts
        present_counts = collections.Counter([g for g, r in zip(guess, guess_result) if r == GuessResult.PRESENT or r == GuessResult.CORRECT])
        for c, v in present_counts.items():
            present_character_counts[c] = max(v, present_character_counts.get(c, 0))
        for i, (letter, result) in enumerate(zip(list(guess), guess_result)):
            if result == GuessResult.CORRECT:
                letter_sets[i] = set()
                known[i] = letter
        for i, (letter, result) in enumerate(zip(list(guess), guess_result)):
            if result == GuessResult.ABSENT:
                if letter not in present_character_counts:
                    for s in letter_sets:
                        if letter in s:
                            s.remove(letter)
                elif letter in letter_sets[i]:
                    # if we guessed the same letter 2 times but there's only one,
                    # one of them will be marked PRESENT while the other is
                    # marked as ABSENT -- even though this second one is
                    # marked as ABSENT here, it just means that it's not at
                    # this location (and a second one doesn't exist)
                    # extrapolates to 3+ as well
                    letter_sets[i].remove(letter)
            elif result == GuessResult.PRESENT and letter in letter_sets[i]:
                letter_sets[i].remove(letter)

    def _find_invalidated_solutions(self, valid_solutions, known, letter_sets, present_character_counts):
        regex_parts = [''] * len(known)
        for i in range(len(known)):
            if known[i] is not None:
                regex_parts[i] = known[i]
            else:
                regex_parts[i] = '[' + ''.join(letter_sets[i]) + ']'
        regex = re.compile(''.join(regex_parts))
        to_remove = []
        for soln in valid_solutions:
            if not re.match(regex, soln):
                to_remove.append(soln)
                # print(f'{soln} did not match regex {regex}')
            else:
                char_counts = dict(collections.Counter(list(soln)))
                for c in present_character_counts:
                    if c not in char_counts or char_counts[c] < present_character_counts[c]:
                        # print(f'{soln} did not contain enough {c}; required {self.present_character_counts[c]}')
                        to_remove.append(soln)
                        break
        return to_remove

    def suggest_guess(self):
        if len(self.past_guesses) == 0 and self.cached_guess is not None and len(self.cached_guess) > 0:
            return self.cached_guess
        if len(self.valid_solutions) == 1:
            return next(iter(self.valid_solutions))

        def key(guess):
            removed = 0
            for soln in self.valid_solutions:
                guess_result = Wordle.check(guess, soln)
                known = copy.copy(self.known)
                letter_sets = copy.deepcopy(self.letter_sets)
                present_character_counts = copy.deepcopy(self.present_character_counts)
                self._update_in_place(guess, guess_result, known, letter_sets,
                                      present_character_counts)
                removed += len(self._find_invalidated_solutions(
                    self.valid_solutions,
                    known,
                    letter_sets,
                    present_character_counts,
                ))
            # print(f'Guess {guess} removed {removed} solutions')
            return removed

        return max(
            itertools.chain(np.random.choice(tuple(self.valid_guesses), self.num_guesses), self.valid_solutions),
            key=key)


def run_game(hard_mode, max_attempts, num_guesses, first_guess, valid_solutions, valid_guesses, word):
    wordle = Wordle(word, valid_solutions, valid_guesses, max_attempts=max_attempts)
    solver = WordleSolver(
        wordle.num_letters(),
        valid_solutions,
        valid_guesses,
        cached_first_guess=first_guess,
        hard_mode=hard_mode,
        num_guesses=num_guesses
    )
    guesses = []
    while True:
        guess = solver.suggest_guess()
        result, outcome = wordle.guess(guess)
        guesses.append(f"({len(solver.valid_solutions)} remaining) {guess} {GuessResult.format_result(result)}")
        solver.update(guess, result)
        if outcome is not None:
            outcome_str = 'Won' if outcome == Outcome.WIN else 'Lost'
            outcome_file = sys.stdout if outcome == Outcome.WIN else sys.stderr
            print(
                f'{outcome_str} game with word {word}; guessed {" -> ".join(guesses)}',
                file=outcome_file
            )
            return outcome, len(guesses)


def simulate(hard_mode, max_attempts, num_guesses, first_guess, valid_solutions, valid_guesses):
    fn = functools.partial(
        run_game,
        hard_mode,
        max_attempts,
        num_guesses,
        first_guess,
        valid_solutions,
        valid_guesses
    )

    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(pool.map(fn, valid_solutions))
        avg_guesses = sum((r[1] for r in results)) / len(results)
        print(f"Used {avg_guesses:.2f} guesses on average.")
        win_rate = len(list(filter(lambda r: r[0] == Outcome.WIN, results))) / len(results) * 100.0
        print(f"Win rate: {win_rate:.2f}%")


def interactive(hard_mode, max_attempts, num_guesses, first_guess, valid_solutions, valid_guesses):
    num_letters = len(next(iter(valid_solutions)))
    solver = WordleSolver(
        num_letters,
        valid_solutions,
        valid_guesses,
        cached_first_guess=first_guess,
        hard_mode=hard_mode,
        num_guesses=num_guesses
    )
    replacements = {
        "g": GuessResult.CORRECT,
        "y": GuessResult.PRESENT,
        "b": GuessResult.ABSENT,
    }

    for _ in range(max_attempts):
        print(f"{len(solver.valid_solutions)} solutions remaining")
        next_guess = solver.suggest_guess()
        print(f'Suggested next guess: {next_guess.upper()}')
        guess = input("Input your guess:").strip().lower()
        results = input("Input the results (Y = yellow, G = green, B = blank):").strip().lower()
        if results == ''.join(["g"] * num_letters):
            print("You won!")
            break
        guess_results = [replacements[c] for c in list(results)]
        solver.update(guess, guess_results)

    print("You lost :(")


@click.command()
@click.option("--run-mode", type=click.Choice(["interactive", "simulate"]), default="interactive",
              help="Run in interactive mode (default) or simulate all possible games")
@click.option("--hard-mode", type=click.BOOL, default=False, help="Run games in hard mode")
@click.option("--max-attempts", type=click.INT, default=6, help="Maximum number of attempts in a game")
@click.option("--num-guesses", type=click.INT, default=1000, help="Number of guesses to check while searching for optimum")
@click.option("--words-file", type=click.Path(), default="words.json", help="File containing valid solutions and guesses")
@click.option("--first-guess", type=click.STRING, default="soare",
            help="Default first guess to make. Leave blank for random. (Warning: this will take a long time)")
def main(run_mode, hard_mode, max_attempts, num_guesses, words_file, first_guess):
    with open(words_file, 'r') as wf:
        words_obj = json.load(wf)
        valid_solutions = words_obj['valid_solutions']
        valid_guesses = words_obj['valid_guesses']
    if run_mode == "interactive":
        interactive(hard_mode, max_attempts, num_guesses, first_guess, valid_solutions, valid_guesses)
    elif run_mode == "simulate":
        simulate(hard_mode, max_attempts, num_guesses, first_guess, valid_solutions, valid_guesses)
    else:
        raise ValueError("Invalid run mode!")


if __name__ == '__main__':
    main()
