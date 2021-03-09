# On Worst-case Regret of Linear Thompson Sampling
This repository contains the code that reproduces the simulations in [our paper.](https://arxiv.org/abs/2006.06790)

## Installation
This project requires python 3.7 or higher. The list of the package dependencies can be found in `requirements.txt`. The
following command installs the required packages.
```shell script
conda install --file requirements.txt
```
## Reproducing the results
By running the following commands, the plots in the paper can be reproduced.
```shell script
PYTHONPATH=src python -m example_1
PYTHONPATH=src python -m example_2 --change mu --mu 1.0 --dim 2000
PYTHONPATH=src python -m example_2 --change rho --rho 1.0 --dim 2000
PYTHONPATH=src python -m example_2 --change dim --mu 0.1 --n-value 18
PYTHONPATH=src python -m experiments
```
Each script also accepts a set of parameters used in the experiment which can be found using `--help` argument. For
example, running
```shell script
PYTHONPATH=src python -m example_1 --help
```
yields
```
usage: example_1.py [-h] [--n-iter N_ITER] [--sigma SIGMA] [--tau TAU]
                    [--seed SEED]

Simulate the first example for TS failure.

optional arguments:
  -h, --help       show this help message and exit
  --n-iter N_ITER  number of iterations
  --sigma SIGMA    prior sd
  --tau TAU        noise sd
  --seed SEED      initial random seed
```
