#! /usr/bin/python3

'''
matches main_process_jaad_256_savehkldata.py. Saves file names that corresponds to the .hkl files.
Written by X. Du 2019/07
'''

import os
import numpy as np
import json
import pickle as pkl
from jaad_settings import *


im_list=[]
source_list = []
counter = 0
sum=0
X_seq_im_filename_small = [x[:] for x in [[None] * nt] * batch_size]   #len(X) = 4, len(X[0])=30
cnt = 0  ##the counter for the first index in X_seq_im_filename_small
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
    num_frames = len(im_list_each)
    im_list_each_split = [im_list_each[i * nt:(i + 1) * nt] for i in range((len(im_list_each) + nt - 1) // nt)]
    im_list_each_split_nt = [im_list_each_split[i] for i in range(len(im_list_each_split)) if len(im_list_each_split[i]) == nt]  ###check size
    sum=sum+len(im_list_each_split_nt)
    ##### The following is correct. Similar to the PredNet generator sequence_start_mode == 'unique', where it saves all frames without overlapping.
    for i in range(len(im_list_each_split_nt)):
        print('==Processing sequence #' + str(i) + ' out of '+ str(len(im_list_each_split_nt)) + '==')
        X_im_filename_small = []
        for j in range(nt):
            im_file = im_list_each_split_nt[i][j]
            X_im_filename_small += [im_file] #(nt,)
        if cnt < batch_size:
            print(cnt)
            X_seq_im_filename_small[cnt] = X_im_filename_small
            cnt = cnt+1
        else:
            print(cnt)
            pkl_name_python2 = os.path.join(X_seq_filename_dir_python2, 'X_seq_filename_{:06d}.pkl'.format(
                counter))
            pkl.dump(X_seq_im_filename_small, open(pkl_name_python2, "wb"), protocol=2)
            pkl_name_python3 = os.path.join(X_seq_filename_dir_python3, 'X_seq_filename_{:06d}.pkl'.format(
                counter))
            pkl.dump(X_seq_im_filename_small, open(pkl_name_python3, "wb"), protocol=3)

            print("saved")
            counter = counter + 1
            X_seq_im_filename_small = [x[:] for x in [[None] * nt] * batch_size]  # len(X) = 4, len(X[0])=30
            cnt = 0
            X_seq_im_filename_small[cnt] = X_im_filename_small
            cnt = cnt + 1

print("that's all folks!")