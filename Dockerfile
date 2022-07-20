FROM python:3.7

COPY src app/src
COPY setup.py app/setup.py
COPY requirements/prod.txt app/requirements/prod.txt
COPY config app/config
COPY data  app/data
COPY models app/models
COPY tests app/tests

WORKDIR ./app

RUN pip install -r requirements/prod.txt
RUN pip install -e .