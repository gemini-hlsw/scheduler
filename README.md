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

**Optional step:** You may wish to also fork and clone the [lucupy](https://github.com/gemini-hlsw/lucupy) repository, which is the package that contains the model for this project. Otherwise the package dependency will be installed from pypi.

### Create the project environment:

You should create a set of environment variables before running the scheduler, one option is adding the following lines to your `~/.bash_profile` or equivalent:

```shell
export PYTHONPATH=$PYTHONPATH:{path-to-project-base}
export REDISCLOUD_URL redis://<USER>:<PASSWORD>@redis-12725.c261.us-east-1-4.ec2.cloud.redislabs.com:12725
export APP_VERSION dev
```

Please contact some project staff member for the redis `USER` and `PASSWORD`

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

### Updating your local environment

To update your project, first pull the latest changes in your repository, to do so, go to your repository directory root and run

```shell
cd path/to/your/scheduler/repository
git pull
```

You can also update some of the packages used enabling the virtual environment and running the following command, i.e. lucupy version

```shell
pip install lucupy -U
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
