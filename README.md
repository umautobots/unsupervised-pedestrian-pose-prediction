# Unsupervised Pedestrian Pose Prediction:
**A deep predictive coding network-based approach for autonomous vehicle perception**

by _Xiaoxiao Du, Ram Vasudevan, and Matthew Johnson-Roberson_ at [`UM FCAV`](https://fcav.engin.umich.edu/)


[[`IEEEXplore (RA-M)`](https://ieeexplore.ieee.org/document/9042808)]

[[`Citation`](#Citation)]

In this repository, we provide the code for our PredNet-based unsupervised pedestrian pose prediction pipeline.

## Dependency

This code was developed using Python 3.6 and the Keras framework (https://keras.io/).

The following codes depend on [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker). Tested on a desktop computer with Intel Xeon 2.10GHz CPU, four NVIDIA TITAN X GPUs and 128 GB memory.

This code uses pre-trained COCO weights for human pose estimation. Please download `mask_rcnn_coco_humanpose.h5` from the releases page (https://github.com/Superlee506/Mask_RCNN_Humanpose/releases) and add it under "./src" folder.

## Demo
### Step 1: Download dataset
We experimented on the [JAAD dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and
the [PedX dataset](http://pedx.io/). We gave detailed instructions using JAAD dataset as an example below.

You can download the JAAD Dataset using the following scripts.
```
./download_JAAD_clips.sh
./split_clips_to_frames.sh
```

In `split_clips_to_frames.sh`Line #8-9, Users may change `CLIPS_DIR` and `FRAMES_DIR` according to their own paths for saving mp4 videos and frames.

In the following, we assume the JAAD dataset is available on a mounted docker volume under path `/mnt/workspace/datasets/JAAD/` and all saved results are under `/mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/`. (See `jaad_settings.py` for more path information. Users may change according to their own settings).


### Step 2: Preprocess

We pre-process JAAD images to be 256x456x3 in size and dump the data into HDF5.

```
docker build -t prednet docker_prednet/
./docker_prednet/run.sh 
root@CONTAINER_ID:/prednet# ./main_process_jaad_256_savehkldata.py 
```

We also save the image file names into HDF5 files.
```
docker build -t prednet_py3 docker_prednet_py3/
./docker_prednet_py3/run.sh
root@CONTAINER_ID:/prednet# ./main_process_jaad_256_savefilename.py
```

### Step 3: Training and testing PredNet models for frame prediction
```
docker build -t prednet docker_prednet/
./docker_prednet/run.sh 
root@CONTAINER_ID:/prednet# ./main_jaad_prednet_train_256.py        ###train PredNet, multi-gpu model. This may take a while depending on the number of training epochs.
root@CONTAINER_ID:/prednet# ./main_jaad_prednet_test_256.py         ###test PredNet and compute MSE and SSIM metrics.
root@CONTAINER_ID:/prednet# ./main_jaad_prednet_test_256_upsample   ###save PredNet frame prediction results as images.
```

### Step 4: Pose prediction

Run OpenPose on both predicted and actual frames.
```
docker build -t openpose:v1.5.0 docker_openpose/
./docker_openpose/run.sh
root@CONTAINER_ID:~/openpose# /root/openpose/build/examples/openpose/openpose.bin --image_dir /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_hat_images/ --display 0 --model_folder /root/openpose/models --write_images /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_hat_images_openpose/ --write_json /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_hat_images_openpose_json/
root@CONTAINER_ID:~/openpose# /root/openpose/build/examples/openpose/openpose.bin --image_dir /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_test_images/ --display 0 --model_folder /root/openpose/models --write_images /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_test_images_openpose/ --write_json /mnt/workspace/datasets/JAAD/JAAD_seq_256x456/weights_100/X_test_images_openpose_json/
```

Run Mask R-CNN (as a non-human noise filter).

```
docker build -t maskrcnn docker_maskrcnn/
./docker_maskrcnn/run.sh
root@CONTAINER_ID:/maskrcnn# ./main_jaad_maskrcnn.py
```

### Step 5: Evaluate pose prediction results

Compute RMSE of joint locations in skeleton poses.

```
docker build -t tf3 docker_prednet_py3/
./docker_prednet_py3/run.sh
root@CONTAINER_ID:/prednet# ./compute_rmse_pose_jaad.py 
```

### Step 6: Multiple-Timestep Prediction (MTP) (optional)
current `extrap_start_time = 10` (given 10 actual frames as input and extrapolate the next 20 frames based on previous predictions).

```
docker build -t prednet docker_prednet/
./docker_prednet/run.sh 
root@CONTAINER_ID:/prednet# ./main_jaad_prednet_train_256_MTP.py    
```
* Note: This saves a MTP fine-tuned model as 'prednet_train_weights-extrap.h5' (See Lines #157-166 in `jaad_settings.py`). After training the MTP model, users may return to Step 3 and load this new fine-tuned model and re-run `main_jaad_prednet_test_256.py` and so on, and compute MSE, SSIM, and RMSE results for MTP frame and pose prediction.

## License

This source code is licensed under the license found in the License.txt file in the root directory of this source tree.

This product is Copyright (c) 2020 X. Du, R. Vasudevan, and M. Johnson-Roberson. All rights reserved.

## <a name="Citation"></a>Citation
Plain Text:
```
X. Du, R. Vasudevan and M. Johnson-Roberson, "Unsupervised Pedestrian Pose Prediction: A Deep Predictive Coding Network-Based Approach for Autonomous Vehicle Perception," in IEEE Robotics & Automation Magazine, vol. 27, no. 2, pp. 129-138, June 2020, doi: 10.1109/MRA.2020.2976313.
```
BibTeX:
```
@ARTICLE{du2020unsupervised,
  author={X. {Du} and R. {Vasudevan} and M. {Johnson-Roberson}},
  journal={IEEE Robotics and Automation Magazine}, 
  title={Unsupervised Pedestrian Pose Prediction: A Deep Predictive Coding Network-Based Approach for Autonomous Vehicle Perception}, 
  year={2020},
  volume={27},
  number={2},
  pages={129-138},}
```

## References:

[The original PredNet code repository](https://github.com/coxlab/prednet) 

[The original Mask R-CNN code repository](https://github.com/matterport/Mask_RCNN) 


