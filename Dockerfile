FROM python:3.11-alpine

# Default app version is development
ARG APP_VERSION="development"

ENV APP_VERSION=$APP_VERSION

WORKDIR /home

# Manage dependencies
COPY ./requirements.txt /home/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

COPY ./scheduler /home/scheduler
COPY ./definitions.py /home/definitions.py

RUN tar -xjf /home/scheduler/services/horizons/data/ephemerides.tar.bz2 -C /home/scheduler/services/horizons/data

ENV PYTHONPATH="${PYTHONPATH}:/scheduler"
ENTRYPOINT [ "python", "/home/scheduler/main.py" ]
