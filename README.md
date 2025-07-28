# Scheduler

![Python version](https://img.shields.io/badge/python-3.10%7C3.11-blue)
![build](https://github.com/gemini-hlsw/Scheduler/actions/workflows/deploy.yml/badge.svg)
![tests](https://github.com/gemini-hlsw/Scheduler/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/gemini-hlsw/scheduler/branch/main/graph/badge.svg?token=15CBFMK3KP)](https://codecov.io/gh/gemini-hlsw/scheduler)

## Project Overview

The Scheduler is an automated tool for the Gemini Observatory, designed to generate and optimize observation schedules as part of the GPP project. It uses the [lucupy](https://github.com/gemini-hlsw/lucupy) library for modeling and supports both standalone and service modes, as well as a Jupyter-based UI for experimentation.

## Features

- Automated scheduling of observations
- GraphQL API for integration with Gemini Program Platform 
- Jupyter notebooks for interactive exploration
- Docker support for easy deployment

## Environment Variables

Before running the scheduler, set the following environment variables:

- `PYTHONPATH`: Should include the project base path.
- `REDISCLOUD_URL`: Redis connection string (contact project staff for credentials).
- `APP_VERSION`: Application version (e.g., `dev`).

Example:
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/scheduler
export REDISCLOUD_URL=redis://<USER>:<PASSWORD>@redis-12725.c261.us-east-1-4.ec2.cloud.redislabs.com:12725
export APP_VERSION=dev
``` 

## Installation

### Local Development

#### Fork and Clone the Repository

If you plan to contribute, first **fork** the repository on GitHub, then clone your fork:

```shell
# Replace <your-username> with your GitHub username
git clone https://github.com/<your-username>/scheduler.git
cd scheduler
```

If you only want to use the scheduler and do not plan to contribute, you can clone the main repository directly:

```shell
git clone https://github.com/gemini-hlsw/scheduler.git
cd scheduler
```

#### Set up a virtual environment

Using `virtualenv`:
```shell
pip install virtualenv
virtualenv --python=python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run the Scheduler

Standalone script:
```shell
python scheduler/scripts/run.py
```

As a service:
```shell
python scheduler/main.py
``` 

### Docker

Build and run the container:
```shell
docker build -t scheduler .
docker run -dp 8000:8000 scheduler
```
Access the GraphQL console at [http://localhost:8000/graphql](http://localhost:8000/graphql).

## Updating & Troubleshooting

To update your local repository and dependencies:
```shell
git pull
pip install -U lucupy
```
If you encounter issues, ensure you have the latest version of `lucupy` and all dependencies.

## Support

For help, contact the project maintainers
