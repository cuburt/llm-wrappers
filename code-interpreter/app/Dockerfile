FROM python:3.10

# Set the working directory to /app
RUN mkdir /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY . .

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install any needed packages specified in requirements.txt
RUN pip3 install streamlit

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
