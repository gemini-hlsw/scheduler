# Scheduler scripts

### `run_greedymax.py`
Run the Scheduler event loop, using GreedyMax to schedule visits during a configurable number of nights
with the desired visibility calculations.

### `download_programs.py`
Downloads program needed for the Scheduler to work. 
DEFAULT_PROGRAMS variable specifies the list of programs the script is going to download
Programs (the .json.gz file) needs to be on the `data/` for the Provider to work properly.

### `odb.extractor_atoms.py`
Script that generates atom for OCS programs. This is Bryan's experimental work and does not use current
mini-model structures. 