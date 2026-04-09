# 1. Prerequisites
The Scheduler supports both 3.10 and 3.11 Python version. Newer versions are not tested yet.
For the list of dependencies check: requirements.txt.

Redis is currently used in the default configuration but can be deactivated.

## 1.2 Ephemerides files storage for Validation

The system connects to [NASA's Horizons](https://ssd.jpl.nasa.gov/horizons/app.html) to handle Non-Sidereal Targets but 
due to the amount of data we store all the ephemerides files for 2018B Semester that are needed to create the whole semester. 
Is possible to skip this step but the performance of the Scheduler would be severely hindered. The files are in a .bz2
compressed file in `/scheduler/scheduler/services/horizons/data/` and it needs `git-lfs` to be cloned from the repo. 
You can install it from [here](https://git-lfs.com/)

# 2. Installation
## 2.1 Local Development
Download the project source:

```shell
$ git clone https://github.com/gemini-hlsw/scheduler.git
```

### 2.1.1 Create the project environment using [uv](https://docs.astral.sh/uv/getting-started/installation/):

<!--
Add the following line to your `~/.bash_profile` or equivalent:
```shell
$ export PYTHONPATH=$PYTHONPATH:{path-to-project-base}
```
-->

Make sure you have an active 3.11 distribution installed on your machine.

uv can be installed using curl:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then in the project directory, execute:

```shell
$ uv sync
```

### 2.1.2 Unzip ephemerides file storage
```shell
$ tar -xjf /scheduler/scheduler/services/horizons/data/ephemerides.tar.bz2
```


## 2.2 Docker

### 2.2.1. Run Docker-compose
If is the first time running the script, it will take some time to build the images.  
```shell
$ docker build -t scheduler .  
$ docker run -dp 8000:8000 scheduler
```

Docker doesn't need to unzip the ephemerides storage file as that process is done in the Dockerfile

# 4. Configuration

The scheduler service needs the following environment variables:

The redis url env name can't change because is how Heroku links the service on the cloud.
``` shell
# URL to the Redis instance
export REDISCLOUD_URL=https::/url/for/redis
# The version of the instance of the scheduler. In local development this value can be anything.
export API_VERSION=1.0
```

### 5. Access the GraphQL Playground
You can access `http://localhost:8000/graphql` to interact with the GraphQL console.

On how to interact with the Playground go to [First Steps](first-steps.md) page to see some examples.
