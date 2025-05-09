# 1. Prerequisites
The Scheduler supports both 3.10 and 3.11 Python version. Newer versions are not tested yet.
For the list of dependencies check: requirements.txt.

Redis is currently used in the default configuration but can be deactivated.
# 2. Installation
## 2.1 Local Development
Download the project source:
Fork the project and then clone into your desired directory.

You may wish to also fork and clone the lucupy repository, which is the package that contains the model for this project.

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
## 2.2 Docker

1. Run Docker-compose. If is the first time running the script, it will take some time to
build the images.  
```shell
$ docker build -t scheduler .  
$ docker run -dp 8000:8000 scheduler
```

2. You can access `http://localhost:8000/graphql` to interact with the GraphQL console. 

# 3. Configuration

The scheduler service needs the following environment variables:

The redis url env name can't change because is how Heroku links the service on the cloud.
```
# URL to the Redis instance
REDISCLOUD_URL=
# The version of the instance of the scheduler. in local development 
# this value can be 
API_VERSION= 
```