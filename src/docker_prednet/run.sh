#!/bin/bash

### run "docker build -t prednet docker_prednet/" first,
### and run "./docker_prednet/run.sh"

docker run -it --rm --gpus all \
  -v `pwd`:/prednet \
  -v fcav:/mnt/fcav \
  -v workspace:/mnt/workspace \
  -w /prednet \
  prednet

# Note: You may need to change the "-v workspace:/mnt/workspace" line to where you store the JAAD and PedX datasets.