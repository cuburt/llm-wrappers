FROM python:3.10

RUN pip install fastapi uvicorn nest-asyncio
RUN mkdir /server
WORKDIR /server
COPY . .
RUN apt-get update && apt-get -y install nano curl
RUN curl -sfL -o docker.tgz "https://download.docker.com/linux/static/stable/x86_64/docker-25.0.4.tgz" && \
  tar -xzf docker.tgz docker/docker --strip=1 --directory /usr/local/bin && \
  rm docker.tgz
RUN cp docker /usr/local/bin