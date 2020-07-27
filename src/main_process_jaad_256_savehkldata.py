#! /usr/bin/python2
'''
Code for downloading and processing the JAAD Dataset and saving sequences in shape(4, 30, 256, 456, 3). 
Batch size=4, length of sequences=30 frames, image size = 256x456x3.
Latest Revision by X. Du 2020/01
'''

import os
import numpy as np
from imageio import imread
import hickle as hkl
from skimage.transform import resize
from jaad_settings import *


im_list=[]
source_list = []
counter = 0
sum=0
X_seq_im_small = np.array([]).reshape((0, nt,) + desired_im_sz)
####X = np.zeros((0,) + desired_im_sz + (3,), np.uint8)
for video_idx in range(346):  # note: video_idx starts with 0 whereas 'video_0001.xml' starts with 0001
    print('=========== Processing video #' + str(video_idx + 1) + ' out of 346 VIDEOS ===========')
    source_list_each = []
    im_list_each = []
    im_dir = frame_data_path_jaad(video_idx)
    files = list(os.walk(im_dir, topdown=False))[-1][-1]
    im_list += [im_dir + f for f in sorted(files)]
    im_list_each += [im_dir + f for f in sorted(files)] #edited: for each video
    source_list += [im_dir] * len(files)
    source_list_each += [im_dir] * len(files)
    X_each = np.zeros((len(im_list_each),) + desired_im_sz + (3,), np.uint8)  #edited: for each video
    num_frames = len(im_list_each)
    im_list_each_split = [im_list_each[i * nt:(i + 1) * nt] for i in range((len(im_list_each) + nt - 1) // nt)]
    im_list_each_split_nt = [im_list_each_split[i] for i in range(len(im_list_each_split)) if len(im_list_each_split[i]) == nt]  ###check size
    sum=sum+len(im_list_each_split_nt)

    ##### The following is similar to the PredNet generator sequence_start_mode == 'unique', where it saves all frames without overlapping. Used in the paper.
    for i in range(len(im_list_each_split_nt)):
        print('==Processing sequence #' + str(i) + ' out of '+ str(len(im_list_each_split_nt)) + '==')
        X_im_small = np.array([]).reshape((0,) + desired_im_sz)
        for j in range(nt):
            im_file = im_list_each_split_nt[i][j]
            im = imread(im_file)
            im_small = resize(im, desired_im_sz)  #(256, 456, 3), already normalized
            X_im_small = np.concatenate((X_im_small,np.expand_dims(im_small,axis=0)), axis=0)  #(30, 256, 456, 3)
        if len(X_seq_im_small) < batch_size:
            print(len(X_seq_im_small))
            X_seq_im_small = np.concatenate((X_seq_im_small, np.expand_dims(X_im_small, axis=0)), axis=0)
        else:
            print(X_seq_im_small.shape)
            hkl.dump(X_seq_im_small, os.path.join(X_seq_im_small_dir, 'X_seq_{:06d}.hkl'.format(
                counter)))
            print("saved")
            counter = counter + 1
            X_seq_im_small = np.array([]).reshape((0, nt,) + desired_im_sz)
            X_seq_im_small = np.concatenate((X_seq_im_small, np.expand_dims(X_im_small, axis=0)), axis=0)


    ##### The following is similar to the PredNet generator sequence_start_mode == 'all', where it saves all frames with overlapping.
    ####  Correct, but takes a long time to run and save all sequences. 
    # for i in range(num_frames-nt):
        #     print('==========Processing Image #' + str(i) + ' out of '+ str(num_frames-nt) + '============')
        #     X_im_small = np.array([]).reshape((0,) + desired_im_sz)
        #     for j in range(nt):
        #         im_file = im_list_each[i+j]
        #         im = imread(im_file)
        #         im_small = resize(im, desired_im_sz)  #(256, 456, 3)
        #         X_im_small = np.concatenate((X_im_small,np.expand_dims(im_small,axis=0)), axis=0)  #(30, 256, 456, 3)
        #     if len(X_seq_im_small)< batch_size:
        #         print(len(X_seq_im_small))
        #         X_seq_im_small = np.concatenate((X_seq_im_small, np.expand_dims(X_im_small, axis=0)), axis=0)
        #     else:
        #         print(X_seq_im_small.shape)
        #         hkl.dump(X_seq_im_small, os.path.join(X_seq_im_small_dir, 'X_seq_{:06d}.hkl'.format(
        #             counter)))
        #         print("saved")
        #         counter = counter + 1
        #         X_seq_im_small = np.array([]).reshape((0, nt,) + desired_im_sz)
        #         X_seq_im_small = np.concatenate((X_seq_im_small, np.expand_dims(X_im_small, axis=0)), axis=0)

print("that's all folks!")