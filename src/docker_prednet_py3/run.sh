#!/bin/bash

### run "docker build -t tf3 docker_prednet_py3/" first,
### and run "./docker_prednet_py3/run.sh"

docker run -it --rm --gpus all \
  -v `pwd`:/prednet \
  -v fcav:/mnt/fcav \
  -v workspace:/mnt/workspace \
  -w /prednet \
  prednet_py3

# Note: You may need to change the "-v workspace:/mnt/workspace" line to where you store the JAAD and PedX datasets.