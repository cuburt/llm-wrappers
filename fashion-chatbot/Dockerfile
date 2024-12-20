FROM --platform=linux/amd64 python:3.10

# Set the working directory to /app
RUN mkdir /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY . .

RUN apt-get -y update; apt-get -y install \
    curl \
    jq \
    ca-certificates

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 uninstall -y transformer-engine
EXPOSE 8080

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]