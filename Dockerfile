FROM python:3.10-buster

WORKDIR /scheduler

# Manage dependencies
COPY ./requirements.txt /scheduler/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /scheduler/requirements.txt
# Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n", "schedule", "/bin/bash", "-c"]

COPY ./app/ /scheduler/app
COPY ./main.py /scheduler/main.py
COPY ./definitions.py /scheduler/definitions.py
COPY ./config.yaml /scheduler/config.yaml
COPY ./mock /scheduler/mock
COPY ./planmanager.py /scheduler/planmanager.py

ENV PYTHONPATH "${PYTHONPATH}:/scheduler"
ENTRYPOINT [ "python", "main.py" ]