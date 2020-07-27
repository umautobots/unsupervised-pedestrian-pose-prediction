#!/usr/bin/python3

### The pre-trained model, "mask_rcnn_coco_humanpose.h5", was downloaded at: https://github.com/Superlee506/Mask_RCNN_Humanpose/releases
#### Part of the code below is based on demo_human_pose.ipynb at https://github.com/Superlee506/Mask_RCNN_Humanpose/
#### Latest Revision X. Du 2020/01
## requirement: numpy, scikit-image, Cython, pycocotools, tensorflow-gpu, keras,IPython,hickle (optional)

import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

import coco
import utils
import model as modellib
import visualize
from model import log
import pickle as pkl
import json
from jaad_settings import *



########################################################################
################ settings for Mask R-CNN ###############################
########################################################################

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(weights_dir, "mylogs")


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load pre-trained weights (assume weights is in the same directory as the scripts.)
print("=======Loading pre-trained weights for Mask R-CNN. If weights not found, download at: https://github.com/Superlee506/Mask_RCNN_Humanpose/releases.========")
model.load_weights("mask_rcnn_coco_humanpose.h5", by_name=True)

# COCO Class names: For human pose task We just use "BG" and "person"
class_names = ['BG', 'person']


def saveMaskRCNNrois(IMAGE_DIR, save_pkl_dir_python2, save_pkl_dir_python3):
    file_names = next(os.walk(IMAGE_DIR))[2]

    for i in range(len(file_names)):
        print("Processing #" +str(i) + ' out of ' + str(len(file_names)))
        filename_choice = file_names[i]
        image = cv2.imread(IMAGE_DIR+filename_choice)
        print(filename_choice)
        #BGR->RGB
        image = image[:,:,::-1]

        # Run detection
        results = model.detect_keypoint([image], verbose=1)
        r = results[0] # for one image

        log("rois",r['rois'])  #shape(NumPeds, 4)
        log("keypoints",r['keypoints']) #shape(NumPeds,17,3)
        log("class_ids",r['class_ids'])
        log("masks",r['masks'])
        log("scores",r['scores']) #shape(NumPeds,), percentage confidence that it is human

        r_rois = r['rois']
        r_keypoints = r['keypoints']  # shape(NumPeds,17,3)
        r_class_ids = r['class_ids']
        r_masks = r['masks']
        r_scores = r['scores']

        json_data = {}
        json_data['rois'] =r_rois
        json_data['keypoints'] = r_keypoints
        json_data['class_ids'] = r_class_ids
        json_data['scores'] = r_scores
        save_pkl_name_python3 = os.path.join(save_pkl_dir_python3, filename_choice[0:-4] + '_r.pkl')
        save_pkl_name_python2 = os.path.join(save_pkl_dir_python2, filename_choice[0:-4] + '_r.pkl')
        pkl.dump(json_data, open(save_pkl_name_python3, "wb"), protocol=3)
        pkl.dump(json_data, open(save_pkl_name_python2, "wb"), protocol=2)
    return 0


if __name__ == "__main__":

    print("=====Running Mask R-CNN for predicted data============")
    # Directory of images to run detection on
    IMAGE_DIR = X_hat_images_dir
    save_pkl_dir_python2 = X_hat_images_MaskRCNN_py2_dir
    save_pkl_dir_python3 = X_hat_images_MaskRCNN_py3_dir
    saveMaskRCNNrois(IMAGE_DIR, save_pkl_dir_python2, save_pkl_dir_python3)

    print("=====Running Mask R-CNN for actual data============")
    # Directory of images to run detection on
    IMAGE_DIR = X_test_images_dir
    save_pkl_dir_python2 = X_test_images_MaskRCNN_py2_dir
    save_pkl_dir_python3 = X_test_images_MaskRCNN_py3_dir
    saveMaskRCNNrois(IMAGE_DIR, save_pkl_dir_python2, save_pkl_dir_python3)


    print("That's all folks!")
