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

# Installation

## Local Development

### Set up a virtual environment

Using `uv`:

```shell
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create virtual environment and install dependencies
uv sync
```

### Run the Scheduler

#### As a service:

```shell
uv run python scheduler/main.py
```

After the scheduler service is started, it will instantiate a web server in the port 8000 by default. The service is meant to be used through the web UI https://github.com/gemini-hlsw/schedule

#### Standalone script:

The standalone script should be able to run a single plan request directly in the terminal and print the output

```shell
uv run python scheduler/scripts/run.py
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
