# ### Dockerfile by UM Ford Center for Autonomous Vehicles (FCAV)

# Run `docker build -t openpose:v1.5.0 openpose` to build the image

# CUDA 10.0 requires driver version >= 410.48
# CUDA 10.1 requires driver version >= 418.39
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# TODO: Test compatibility with 18.04

RUN apt-get update && \
    apt-get -y install libopencv-dev cmake git sudo

RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose /root/openpose && \
    cd /root/openpose && \
    git checkout v1.5.0

# See "Ubuntu Prerequisites"
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/prerequisites.md#ubuntu-prerequisites
RUN bash /root/openpose/scripts/ubuntu/install_deps.sh

WORKDIR /root/openpose/build

# See "OpenPose Python Module and Demo"
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
RUN cmake -DBUILD_PYTHON=ON .. && \
    make -j`nproc` && \
    make install -j`nproc`

WORKDIR /root/openpose
