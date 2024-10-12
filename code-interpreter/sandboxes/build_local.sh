#!/bin/bash

# CONFIG_DIR=sandboxes/server/.devcontainer
# debug "CONFIG_DIR: ${CONFIG_DIR}"
# CONFIG_FILE=devcontainer.json
# debug "CONFIG_FILE: ${CONFIG_FILE}"
# if ! [ -e "$CONFIG_DIR/$CONFIG_FILE" ]; then
#     echo "Folder contains no devcontainer configuration"
#     exit
# fi

# echo "config file: ${CONFIG_FILE}"

# CONFIG=$(cat $CONFIG_DIR/$CONFIG_FILE | grep -v //)
# debug "CONFIG: \n${CONFIG}"

# echo "config: ${CONFIG}"
# cd $CONFIG_DIR

# PORTS=$(echo $CONFIG | jq -r '.forwardPorts | map("-p \(.):\(.)")? | join(" ")')
# debug "PORTS: ${PORTS}"

docker run --privileged -p 8081:8081 --name dind-container -d $(docker build --platform=linux/amd64 -q -t local/dind:latest .)
sleep 30
docker exec dind-container sh -c "cd /sandboxes/voltscript && /bin/sh build.sh"
sleep 30
docker exec dind-container sh -c "cd /sandboxes/javascript && /bin/sh build.sh"
sleep 30
docker exec dind-container sh -c "cd /sandboxes/python && /bin/sh build.sh"
sleep 30
docker exec dind-container sh -c "cd /sandboxes/server && /bin/sh build.sh"

