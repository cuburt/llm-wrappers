FROM python:3.10

RUN mkdir /data
RUN mkdir /server

# copy the requirements file used for dependencies
COPY server /server
COPY data /data

# Set the working directory to /server
WORKDIR /server

RUN apt-get -y update; apt-get -y install \
    curl \
    jq \
    ca-certificates \
    nano

# RUN apt-get -y install nodejs

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 uninstall -y transformer-engine
EXPOSE 8080

CMD uvicorn server:server --reload --port 8080 --host 0.0.0.0
