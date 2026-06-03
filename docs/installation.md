# Prerequisites

The Scheduler supports both 3.10 and 3.11 Python version. Newer versions are not tested yet.
For the list of dependencies check the pyproject.toml in the `backend` folder.

For the Docker installation the Docker daemon must be running on the local machine.

## Env variables

The system relies heavily on environment variables to function. In the source code we provide a `.env.exmaple` to fill up
necessary variables. You can read more about what each variable represent in the [Configuration](configuration.md) page.

## Runtime calculations and ephemerides files

The Scheduler can be configured to work with both visibility calculations done in runtime and with Sight. If the first
option is selected, due to the amount of data we store all the ephemerides files for 2018B Semester that are needed to create the whole semester.
Is possible to skip this step but the performance of the Scheduler would be severely hindered. The files are in a .bz2
compressed file in `/scheduler/scheduler/services/horizons/data/` and it needs `git-lfs` to be cloned from the repo.
You can install it from [here](https://git-lfs.com/)

To unzip you can run:

```shell
$ tar -xjf /scheduler/scheduler/services/horizons/data/ephemerides.tar.bz2
```

# Installation

First clone the monorepo.

```shell
$ git clone https://github.com/gemini-hlsw/scheduler.git
```

> :bulb:
> Remember that if you use runtime visibility you need to use [Git LFS](https://git-lfs.com/)

```bash
git lfs install
git lfs pull
```

After that you have two options using Docker or using a local setting.

## Local installation

To run a local development version first make sure your environment have the appropriate variables already set.
This will change depending on the shell you are using. In bash and zsh you can copy the .env.example file to a new .env
file, fill the secrets in this new .env file and run

```bash
source .env
```

Also, the `PYTHONPATH` environment variable should be properly set to add the `backend` directory of this repository.
Add the following line to your `~/.bash_profile` or add it to the console session:

```shell
$ export PYTHONPATH=$PYTHONPATH:{path-to-project-base} # should end in scheduler/
```

### Weather

If `REALTIME` mode wants to be used a properly running weather service is required.
Check [external-services/weather](external-services/weather.md) for detailed instructions.

### Backend

```bash
cd backend
uv sync --group gpp-dev --no-group gpp-prod
uv run python scheduler/main.py
```

### Frontend

```bash
pnpm install
pnpm frontend dev
```

## Docker installation

Build and start all services:

```bash
docker compose build
docker compose up
```

| Service              | URL                           | Mode         |
| -------------------- | ----------------------------- | ------------ |
| Frontend             | http://localhost              | —            |
| Backend (Validation) | http://localhost:8001/graphql | `VALIDATION` |
| Backend (Realtime)   | http://localhost:8000/graphql | `REALTIME`   |
| Weather              | http://localhost:4000         | —            |

### Updating the GraphQL schema

When the backend schema changes, regenerate `backend/scheduler.graphql` before rebuilding:

```bash
uv run python scripts/export_graphql_schema.py
docker compose build frontend
```
