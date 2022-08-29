FROM continuumio/miniconda3:latest

# copy all the files
COPY . . 

RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "schedule", "/bin/bash", "-c"]
ENV PYTHONPATH "${PYTHONPATH}:/"