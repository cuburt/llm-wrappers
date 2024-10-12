#!/bin/bash

if ! [ -x "$(command -v jq)" ]; then
    printf "\x1B[31m[ERROR] jq is not installed.\x1B[0m\n"
    exit 1
fi
OPTIND=1
VERBOSE=0

while getopts "v" opt; do
    case ${opt} in
        v ) VERBOSE=1 ;;
    esac
done

debug() {
    if [ $VERBOSE == 1 ]; then
        printf "\x1B[33m[DEBUG] ${1}\x1B[0m\n"
    fi
}

WORKSPACE=${1:-`pwd`}
CURRENT_DIR=${PWD##*/}
echo "Using workspace ${WORKSPACE}"

CONFIG_DIR=./.devcontainer
debug "CONFIG_DIR: ${CONFIG_DIR}"
CONFIG_FILE=devcontainer.json
debug "CONFIG_FILE: ${CONFIG_FILE}"
if ! [ -e "$CONFIG_DIR/$CONFIG_FILE" ]; then
    echo "Folder contains no devcontainer configuration"
    exit
fi

echo "config file: ${CONFIG_FILE}"

CONFIG=$(cat $CONFIG_DIR/$CONFIG_FILE | grep -v //)
debug "CONFIG: \n${CONFIG}"

echo "config: ${CONFIG}"
cd $CONFIG_DIR

DOCKER_FILE=$(echo $CONFIG | jq -r .dockerFile)
DOCKER_IMAGE_HASH=$(echo $CONFIG | jq -r .image)

if [ "$DOCKER_FILE" == "null" ]; then
    DOCKER_FILE=$(echo $CONFIG | jq -r .build.dockerfile)
    echo "dockerfile: ${DOCKER_FILE}"
fi

DOCKER_FILE=$(readlink -f $DOCKER_FILE)
debug "DOCKER_FILE: ${DOCKER_FILE}"
echo "DOCKER_FILE: ${DOCKER_FILE}"
if ! [ -e $DOCKER_FILE ]; then
  echo "Can not find dockerfile ${DOCKER_FILE}"
  exit
fi

echo "step 1"
SANDBOX_NAME=$(echo $CONFIG | jq -r .name)
echo "$SANDBOX_NAME"

REMOTE_USER=$(echo $CONFIG | jq -r .remoteUser)
debug "REMOTE_USER: ${REMOTE_USER}"
if ! [ "$REMOTE_USER" == "null" ]; then
    REMOTE_USER="-u ${REMOTE_USER}"
fi

ARGS=$(echo $CONFIG | jq -r '.build.args | to_entries? | map("--build-arg \(.key)=\"\(.value)\"")? | join(" ")')
debug "ARGS: ${ARGS}"

SHELL=$(echo $CONFIG | jq -r '.settings."terminal.integrated.shell.linux"')
debug "SHELL: ${SHELL}"

PORTS=$(echo $CONFIG | jq -r '.forwardPorts | map("-p \(.):\(.)")? | join(" ")')
debug "PORTS: ${PORTS}"

ENVS=$(echo $CONFIG | jq -r '.remoteEnv | to_entries? | map("-e \(.key)=\(.value)")? | join(" ")')
debug "ENVS: ${ENVS}"

WORK_DIR="/workspace"
debug "WORK_DIR: ${WORK_DIR}"

MOUNT="${MOUNT} --mount type=bind,source=${WORKSPACE},target=${WORK_DIR}"
debug "MOUNT: ${MOUNT}"


if [ "$DOCKER_IMAGE_HASH" == "null" ]; then
  echo "No image specified. Building and starting container..."
  DOCKER_IMAGE_HASH=$(docker build --platform linux/amd64 -t $SANDBOX_NAME -f $DOCKER_FILE $ARGS .)
fi

debug "DOCKER_IMAGE_HASH: ${DOCKER_IMAGE_HASH}"
echo "docker run --platform linux/amd64 --privileged=true -u 0 -it $PORTS $ENVS $MOUNT -w $WORK_DIR --name $SANDBOX_NAME -d $DOCKER_IMAGE_HASH"
docker run --platform linux/amd64 --privileged=true -u 0 -it $PORTS $ENVS $MOUNT -w $WORK_DIR --name $SANDBOX_NAME -d $DOCKER_IMAGE_HASH