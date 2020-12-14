#!/usr/bin/env bash

IMG=libtorch_tutorial:latest
SRC_DIR=$(cd $(dirname $0);pwd)
DIRNAME=$(basename $(dirname ${0}))
docker run --rm -it \
    -w /root/${DIRNAME} \
    -v ${PWD}:/root/${DIRNAME} \
    -v ~/.Xauthority:/root/.Xauthority \
    --net host \
    ${IMG} \
    bash
