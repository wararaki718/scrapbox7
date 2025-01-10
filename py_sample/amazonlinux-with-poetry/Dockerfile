FROM amazonlinux:2023

RUN dnf install python3.11 python-pip3.11 -y
RUN ln -s -f /usr/bin/python3.11 /usr/local/bin/python

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /app
COPY src /app

RUN pip3.11 install poetry
RUN poetry install

ENTRYPOINT [ "poetry", "run", "batch-runner" ]
