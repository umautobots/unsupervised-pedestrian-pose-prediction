# Parameter and Path settings for training and testing PredNet Pose Predcition

#### Latest Revision X. Du 2020/01

import os 
import subprocess
import numpy as np
import hickle as hkl

random_seed = 100 # random seed (for randomly shuffle train/val/test data), can be changed

############################################################################################
##############Path Settings (Can be changed by users) ######################################
############################################################################################
ROOT_DIR = '/mnt/workspace/datasets/JAAD/'  #ROOT_DIR
RESULTS_DIR = '/mnt/workspace/datasets/JAAD/JAAD_seq_256x456/'  #RESULTS_DIR

def frame_data_path_jaad(video_idx):
	frame_data_path = ROOT_DIR +'JAAD_images/video_{:04d}/'.format(video_idx + 1) 
	return frame_data_path

##### path for saving 30-frame sequences
X_seq_im_small_dir = RESULTS_DIR + 'hkl_data/' # save 256x456x3 images
if not os.path.exists(X_seq_im_small_dir): os.mkdir(X_seq_im_small_dir)

##### path for saving filenames for the 30-frame sequences
hkl_filenames_DIR = RESULTS_DIR + 'hkl_filenames/'
if not os.path.exists(hkl_filenames_DIR): os.mkdir(hkl_filenames_DIR)
X_seq_filename_dir_python2 = hkl_filenames_DIR + 'python2/'
X_seq_filename_dir_python3 = hkl_filenames_DIR + 'python3/'
if not os.path.exists(X_seq_filename_dir_python2): os.mkdir(X_seq_filename_dir_python2)
if not os.path.exists(X_seq_filename_dir_python3): os.mkdir(X_seq_filename_dir_python3)

##### path for saving PredNet weights
weights_dir = RESULTS_DIR + 'weights_' + str(random_seed) + '/'
if not os.path.exists(weights_dir): os.mkdir(weights_dir)

save_model = True  # if True, weights will be saved
weights_file = weights_dir + 'prednet_train_weights.h5'  # where weights will be saved
json_file_name = weights_dir + 'prednet_train_model.json'


##### path for saving PredNet test results: the ``up-sampled'' (dpi=1000) saved frame prediction 
X_test_images_dir = weights_dir + 'X_test_images/'
X_hat_images_dir = weights_dir + 'X_hat_images/'
if not os.path.exists(X_test_images_dir): os.mkdir(X_test_images_dir)
if not os.path.exists(X_hat_images_dir): os.mkdir(X_hat_images_dir)


##### path for saving OpenPose and Mask R-CNN results

X_hat_images_MaskRCNN_dir        = weights_dir + 'X_hat_images_MaskRCNN/'
X_test_images_MaskRCNN_dir       = weights_dir + 'X_test_images_MaskRCNN/'
X_hat_images_openpose_json_dir   = weights_dir + 'X_hat_images_openpose_json/'
X_test_images_openpose_json_dir  = weights_dir + 'X_test_images_openpose_json/'
X_hat_images_openpose_dir        = weights_dir + 'X_hat_images_openpose/'
X_test_images_openpose_dir       = weights_dir + 'X_test_images_openpose/'
X_hat_images_MaskRCNN_py2_dir    = weights_dir + 'X_hat_images_MaskRCNN/python2'
X_hat_images_MaskRCNN_py3_dir    = weights_dir + 'X_hat_images_MaskRCNN/python3'
X_test_images_MaskRCNN_py2_dir   = weights_dir + 'X_test_images_MaskRCNN/python2'
X_test_images_MaskRCNN_py3_dir   = weights_dir + 'X_test_images_MaskRCNN/python3'


if not os.path.exists(X_hat_images_MaskRCNN_dir):          os.mkdir(X_hat_images_MaskRCNN_dir)
if not os.path.exists(X_test_images_MaskRCNN_dir):         os.mkdir(X_test_images_MaskRCNN_dir)
if not os.path.exists(X_hat_images_openpose_json_dir):     os.mkdir(X_test_images_openpose_json_dir)
if not os.path.exists(X_test_images_openpose_json_dir):    os.mkdir(X_test_images_openpose_json_dir)
if not os.path.exists(X_hat_images_openpose_dir):          os.mkdir(X_hat_images_openpose_dir)
if not os.path.exists(X_test_images_openpose_dir):         os.mkdir(X_test_images_openpose_dir)
if not os.path.exists(X_hat_images_MaskRCNN_py2_dir):      os.mkdir(X_hat_images_MaskRCNN_py2_dir)
if not os.path.exists(X_hat_images_MaskRCNN_py3_dir):      os.mkdir(X_hat_images_MaskRCNN_py3_dir)
if not os.path.exists(X_test_images_MaskRCNN_py2_dir):     os.mkdir(X_test_images_MaskRCNN_py2_dir)
if not os.path.exists(X_test_images_MaskRCNN_py3_dir):     os.mkdir(X_test_images_MaskRCNN_py3_dir)



############################################################################################
##############Parameter Settings for PredNet experiments ###################################
############################################################################################

desired_im_sz = (256, 456, 3) #save 256x456x3 images for prednet
# Training parameters
nt = 30  #30-frame sequence length, fixed
batch_size = 4  #batch size, fixed

shuffle = True #True or False -- if True, randomly shuffle input sequences
train_perc = 0.3  #85% sequences for training, can be changed
val_perc = 0.3   #5% sequences for validation, can be changed
nb_epoch = 150 # number of epochs, can be changed

num_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')  #number of gpus, automatically detected
np.random.seed(random_seed)

###################################################################################################
################ utils for loading train/val/test images in PredNet ###############################
###################################################################################################
def imageLoader(files, batch_size):
    # ###Modified: load batches. Here the batchsize input =1, since the input sequences are already of shape (4,30,256,456,3).
    L = len(files)
    while True:   
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            filename = files[batch_start]
            X = hkl.load(filename)
            Y=  np.zeros((4,), np.float32)  #batch_size
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size



file_dir = X_seq_im_small_dir
fileList = []
files = list(os.walk(file_dir, topdown=False))[-1][-1]
fileList += [file_dir + f for f in sorted(files)]
TotalNumTrackSeq = len(fileList)
train_size = int(TotalNumTrackSeq * train_perc)
if (val_perc > 0.0):
    val_size = int(TotalNumTrackSeq * val_perc)
    test_size = TotalNumTrackSeq - train_size - val_size
    # print(train_size, val_size, test_size)
else:
    test_size = TotalNumTrackSeq - train_size
    # print(train_size, test_size)

if shuffle:
    DatasetSeqIdx_rp = np.random.permutation(TotalNumTrackSeq)
else:
    DatasetSeqIdx_rp = range(TotalNumTrackSeq)

DatasetSeqIdx_rp_train = DatasetSeqIdx_rp[0:train_size]
if (val_perc > 0.0):
    DatasetSeqIdx_rp_val = DatasetSeqIdx_rp[train_size:train_size + val_size]
    DatasetSeqIdx_rp_test = DatasetSeqIdx_rp[train_size + val_size:TotalNumTrackSeq]
else:
    DatasetSeqIdx_rp_test = DatasetSeqIdx_rp[train_size:TotalNumTrackSeq]

fileList_train = [fileList[i] for i in DatasetSeqIdx_rp_train]
if (val_perc > 0.0):
    fileList_val = [fileList[i] for i in DatasetSeqIdx_rp_val]
    fileList_test = [fileList[i] for i in DatasetSeqIdx_rp_test]
else:
    fileList_test = [fileList[i] for i in DatasetSeqIdx_rp_test]

# print(len(fileList_train))
# print(len(fileList_val))
# print(len(fileList_test))



###################################################################################################
################ Multiple Timestep Prediction (MTP) settings for PredNet ##########################
###################################################################################################
##### MTP prediction extrapolation. Note: this requires running and loading the trained non-MTP weights first
MTP_Flag = True # True or False -- if True, run multiple-timestep prediction (extrapolation)
extrap_start_time = 10 #starting #extrap_start_time, use previous prediction as input in PredNet, can be changed

##### path for saving PredNet weights
weights_dir_MTP = RESULTS_DIR + 'weights_' + str(random_seed) + '_MTP/'
if not os.path.exists(weights_dir): os.mkdir(weights_dir)

save_model_MTP = True  # if True, weights will be saved
extrap_weights_file = weights_dir_MTP + 'prednet_train_weights-extrap.h5'  # where weights will be saved
extrap_json_file = weights_dir_MTP + 'prednet_train_model-extrap.json'

