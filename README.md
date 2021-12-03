Scheduler

The automated Scheduler for Gemini Observatory.

Note that you need the following libraries to run this:
* astropy
* matplotlib
* openpyxl
* pytz
* requests
* tabulate
* joblib
* tqdm
* numpy >= 1.21.4 (for numpy typing)

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
