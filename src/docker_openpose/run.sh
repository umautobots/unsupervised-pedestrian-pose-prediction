#! /bin/bash

sudo docker run -it --gpus all \
	-v fcav:/mnt/fcav \
	-v workspace:/mnt/workspace \
	openpose:v1.5.0  \
	/bin/bash

# Note: You may need to change the "-v workspace:/mnt/workspace" line to where you store the JAAD and PedX datasets.