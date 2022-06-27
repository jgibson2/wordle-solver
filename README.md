# Overview

This is a wordle solver. There are two modes:

```shell
python3 solver.py
```
for an interactive solver and

```shell
python3 solver.py --run-mode simulate
```
to simulate all possible Wordle games and report the average number of guesses and win rate.

# Usage
```
Usage: solver.py [OPTIONS]

Options:
  --run-mode [interactive|simulate]
                                  Run in interactive mode (default) or
                                  simulate all possible games
  --hard-mode BOOLEAN             Run games in hard mode
  --max-attempts INTEGER          Maximum number of attempts in a game
  --num-guesses INTEGER           Number of guesses to check while searching
                                  for optimum
  --words-file PATH               File containing valid solutions and guesses
  --first-guess TEXT              Default first guess to make. Leave blank for
                                  random. (Warning: this will take a long
                                  time)
  --help                          Show this message and exit.
```

# How It Works

The core idea of the system is quite simple: greedily make a guess that eliminates
the maximal number of possibilities until only one remains. In practice, this takes several steps.

### The First Guess
The optimal first guess can be determined by searching the entire space of guesses for the one that
eliminates the most solutions, in expectation. Because each new game is independent, this can be used
for every game. The optimal (one-step lookahead) guess is **soare**, which is a type of young hawk, but you can set the tool
to use a different word initially, or let it choose one itself (but be warned that this will take a long time).

Note: multi-step lookahead determined the optimal first word is **salet**, a kind of helmet.

### Subsequent Guesses
After the first guess, the algorithm will filter out potential solutions using the information gained
from each round:
* Any solutions without all CORRECT letters in the proper positions are removed
* Any solutions with letters marked ABSENT (and not PRESENT) are removed
* Any leftover solutions without all CORRECT and PRESENT letters (in the proper multiplicities) are also removed

Then, a large number of guesses are sampled from the valid guess list (default all) and combined with all remaining valid solutions,
and the one that gives the largest possible amount of information is returned as the 
next guess. When the number of remaining solutions equals one, it is returned.

# Results

The algorithm averages 3.53 guesses per game in normal mode and has a win rate of 100%. In hard mode, the algorithm
averages 3.67 guesses per game and has a win rate of 99.61%. 
