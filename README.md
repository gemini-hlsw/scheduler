# Scheduler

![Python version](https://img.shields.io/badge/python-3.10%7C3.11-blue)
![build](https://github.com/gemini-hlsw/Scheduler/actions/workflows/deploy.yml/badge.svg)
![tests](https://github.com/gemini-hlsw/Scheduler/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/gemini-hlsw/scheduler/branch/main/graph/badge.svg?token=15CBFMK3KP)](https://codecov.io/gh/gemini-hlsw/scheduler)

This is the automated Scheduler for Gemini Observatory, part of the GPP project.

For the list of dependencies check: `requirements.txt`.

## How to Install (Local Development)

**Note:** These instructions assume you are using Mac OS X or Linux.

### Download the project source:

Fork the project and then clone into your desired directory.

You may wish to also fork and clone the [lucupy](https://github.com/gemini-hlsw/lucupy) repository, which is the
package that contains the model for this project.

### Create the project environment:

<!--
Add the following line to your `~/.bash_profile` or equivalent:
```shell
$ export PYTHONPATH=$PYTHONPATH:{path-to-project-base}
```
-->

#### Using [virtualenv](https://virtualenv.pypa.io/en/latest/):

Make sure you have an active Python 3.10 or 3.11 distribution installed on your machine.

virtualenv can be installed using pip:
```shell
$ pip install virtualenv
```

Then in the project directory, execute:

```shell
$ virtualenv --python=/path/to/python_executable venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

#### Using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Anaconda](https://www.anaconda.com):

In the project directory, execute:

```shell
$ conda env create -f environment.yml
$ conda activate scheduler
```

### Executing the Scheduler

#### Standalone script

To run the scheduler as a standalone script, execute:

```shell
$ python scheduler/scripts/run.py
```

If you have performed the installation correctly, you should see logging messages, and an output of a plan, followed
by the message `DONE`.

#### Service

To run the scheduler as a service, execute:

```shell
$ python scheduler/main.py
```

#### Jupyter notebooks

We offer Jupyter notebooks using a [Mercury](https://github.com/mljar/mercury) user interface to test the scheduler.
This can be launched on `localhost:8000` as follows:

```shell
$ cd demo
$ mercury run
```

This should open a tab in your active browser and show the notebooks.

If the startup complains about a missing `allauth` package, install this with:

```shell
$ pip install django-allauth
```

## How to Install (Docker)

1. Run Docker-compose. If is the first time running the script, it will take some time to
build the images.  
```shell
$ docker build -t scheduler .  
$ docker run -dp 8000:8000 scheduler
```

2. You can access `http://localhost:8000/graphql` to interact with the GraphQL console. 

## Troubleshooting

The most likely cause of issues during execution is that changes have been made to the [lucupy](https://github.com/gemini-hlsw/lucupy)
project and an update of the package is necessary. This can be done with:

```shell
$ pip install -U lucupy
```
