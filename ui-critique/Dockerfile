FROM --platform=linux/amd64 us-east4-docker.pkg.dev/hclsw-gcp-xai/img-repo/base-image:transformers-torch-noncuda2.4.0

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
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8080", "--server.address=0.0.0.0"]
