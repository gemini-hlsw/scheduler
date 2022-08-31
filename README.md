Scheduler

This is the automated Scheduler for Gemini Observatory, part of the GPP project.

It currently is supported to run on Python >= 3.9.

For the list of dependencies check: `environment.yml`

## How to Install (Local Development)

**Note:** These instructions assume you are using Mac OS X or Linux.

1. Fork the project and then clone into your desired directory.

2. Add the following line to your `~/.bash_profile` or equivalent:
```shell
$ export PYTHONPATH=$PYTHONPATH:{path-to-project-base}
```

2. Install either conda or Anaconda on your machine.
* **Anaconda:** https://www.anaconda.com
* **conda:** https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Go into the base path for the project and run the following to install the conda environment:
```shell
$ conda env create -f environment.yml
$ conda activate greedymax-env
```

To run the scheduler do:
```shell
$ cd scripts
$ python run_scheduler.py
```

If you have performed the installation correctly, you should see some basic
output with no errors and the message DONE.


## How to Install (Docker)

1. Run Docker-compose. If is the first time running the script, it would take some time to
build the Images.  
```shell
$ docker compose up 
```

2. You can access `http://localhost:8000/graphql` to interact with the GraphQL console. 


## Notes
* For Collector, look into `cached_propperty`.
