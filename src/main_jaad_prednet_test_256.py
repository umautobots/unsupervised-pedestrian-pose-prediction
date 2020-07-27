#! /usr/bin/python2


#### Test PredNet on the pre-processed JAAD (4,30,256,456,3) sequences, on multiple GPUs.
#### Based on kitti_evaluate.py in PredNet original code release.
#### Latest Revision X. Du 2020/01

import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
import os
from six.moves import cPickle
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from prednet import PredNet
import subprocess
from keras.models import model_from_json

from jaad_settings import *
np.random.seed(random_seed)


print("=========================testing PredNet on JAAD=====================")  #

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
train_model = model_from_json(loaded_model_json, custom_objects = {'PredNet': PredNet})

# load weights into new model
train_model.load_weights(weights_file)
print("Loaded model from disk")


# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)


X_test = hkl.load(fileList_test[0])
for i in range(1,len(fileList_test)):
    print("==Processing X_test #" + str(i) + " out of " + str(len(fileList_test)) + "==")
    X_each = hkl.load(fileList_test[i])
    X_test= np.concatenate((X_test,X_each),axis=0)



X_hat = test_model.predict(X_test, batch_size) ##(276, 30, 256, 456, 3)
#print(X_hat.shape)

if save_model: ### this step saves the concatenated X_hat (prediction) and X_test (actual), may take a while
    hkl.dump(X_hat, os.path.join(weights_dir, 'X_hat.hkl'))  
    hkl.dump(X_test, os.path.join(weights_dir, 'X_test.hkl'))

# Compare MSE and SSIM of PredNet predictions versus copying last frame
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )

print(mse_model)
print(mse_prev)


####from skimage.metrics import structural_similarity as ssim ###require skimage version>0.16
from skimage.measure import compare_ssim as ssim

ssim_noise=np.empty((len(X_test),nt-1))
ssim_prev=np.empty((len(X_test),nt-1))
for i in range(len(X_test)):
    print("==Computing SSIM for #"+str(i) + ' out of ' + str(len(X_test)) + '==')
    for j in range(1,nt):
        ssim_noise[i,j-1] = ssim(X_test[i, j] , X_hat[i, j],multichannel=True)
        ssim_prev[i,j-1] = ssim(X_test[i, j] , X_test[i, j-1],multichannel=True)

print(np.mean(ssim_noise))
print(np.mean(ssim_prev))

#### the following is grammatically correct, but for this dataset may be out of memory
# ssim_noise = ssim(X_test[:, 1:] , X_hat[:, 1:],
#                   data_range=X_hat[:, 1:].max() - X_hat[:, 1:].min(),multichannel=True)

# ssim_prev = ssim(X_test[:, :-1], X_test[:, 1:],
#                   data_range=X_test[:, 1:].max() - X_test[:, 1:].min())


print("that's all folks!")
