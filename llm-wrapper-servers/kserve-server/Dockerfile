# Python image to use.
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

RUN apt-get -y update; apt-get -y install \
     curl \
     jq \
     ca-certificates \
     nano

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PORT 8080

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . /app

# Run app.py when the container launches
ENTRYPOINT ["python3","-u","models.py"]
