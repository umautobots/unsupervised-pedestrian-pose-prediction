#!/bin/bash

docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=shore.engin.umich.edu,rw \
  --opt device=:/volume2/fcav \
  fcav

docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=shore.engin.umich.edu,rw \
  --opt device=:/volume1/UMFORDAVDATA \
  ngv

docker volume create --driver local \
  --opt type=nfs \
  --opt o=addr=seal.engin.umich.edu,rw \
  --opt device=:/mnt/pool1/workspace \
  workspace