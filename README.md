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
export REDISCLOUD_URL=redis://<USER>:<PASSWORD>@redisserver.ec2.cloud.redislabs.com:12345
export APP_VERSION=dev
``` 

## Installation

### Local Development

#### Clone the Repository and Create a Feature Branch

> **Note:**  
> To ensure GitHub Actions secrets (such as `REDISCLOUD_URL`) are available for testing and CI, all contributions should be made from branches within the main repository, **not from forks**.

1. **Clone the main repository:**
   ```shell
   git clone https://github.com/gemini-hlsw/scheduler.git
   cd scheduler
   ```

2. **Create a new feature branch:**
   ```shell
   git checkout -b your-feature-branch
   ```

3. **Work on your changes and commit as usual.**

4. **Rebase your branch with the latest main branch before pushing:**
   ```shell
   git fetch origin
   git rebase origin/main
   ```

5. **Push your branch to the main repository:**
   ```shell
   git push origin your-feature-branch
   ```

6. **Open a Pull Request** from your feature branch to `main` in the [main repository](https://github.com/gemini-hlsw/scheduler).

> **Do not use the GitHub fork workflow.**  
> Opening pull requests from forks will not have access to required repository secrets, and CI/CD workflows may fail.

#### Set up a virtual environment

Using `uv`:
```shell
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create virtual environment and install dependencies
uv sync

#### Run the Scheduler

Standalone script:
```shell
uv run python scheduler/main.py
```

As a service:
```shell
uv run python scheduler/main.py
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
