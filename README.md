Scheduler

The automated Scheduler for Gemini Observatory.

It currently is supported to run on Python 3.9.x.

Note that you need the following libraries to run this:
* astropy
* hypothesis
* matplotlib
* more_itertools
* numpy (>= 1.21.4 for numpy typing)
* openpyxl
* pytest
* pytz
* tabulate
* tqdm

## Install

To install conda environment run :

```
$ conda env create -f environment.yml
$ conda activate greedymax-env
```

To run the scheduler do: 

```
$ python solver_greedy_max.py
```
