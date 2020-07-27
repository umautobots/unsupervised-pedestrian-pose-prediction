#!/bin/bash
docker run -it --rm --gpus all \
-p 8888:8888 -p 6006:6006 \
  -v `pwd`:/maskrcnn \
  -v fcav:/mnt/fcav \
  -v workspace:/mnt/workspace \
  -w /maskrcnn \
maskrcnn

## Based onhttps://hub.docker.com/r/waleedka/modern-deep-learning/  working copy
## Note: You may need to change the "-v workspace:/mnt/workspace" line to where you store the JAAD and PedX datasets.