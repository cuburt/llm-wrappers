FROM --platform=linux/amd64 python:3.10

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN apt-get -y update; apt-get -y install \
  curl \
  jq \
  ca-certificates \
  nano \
  npm
RUN pip3 install -r requirements.txt

ARG GEMINI_APIKEY
ENV GEMINI_APIKEY=${GEMINI_APIKEY}

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8080", "--server.address=0.0.0.0"]
