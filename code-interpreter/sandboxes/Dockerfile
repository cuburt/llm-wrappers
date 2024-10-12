FROM docker:dind

# Set the working directory to /server
RUN mkdir /sandboxes

# copy the requirements file used for dependencies
COPY . /sandboxes

RUN apk add jq nano
ENV USERNAME=cuburt.balanon@hcl.com
RUN cat /sandboxes/voltscript/.password.txt | docker login https://hclcr.io --username $USERNAME --password-stdin
RUN apk add curl
EXPOSE 8081