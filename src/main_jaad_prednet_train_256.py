#! /usr/bin/python2


#### Train PredNet on the pre-processed JAAD (4,30,256,456,3) sequences, on multiple GPUs.
#### Based on kitti_train.py in PredNet original code release.
#### Latest Revision X. Du 2020/01

import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.models import Model
from keras.layers import Input, Dense, Flatten,LSTM,TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from prednet import PredNet
from jaad_settings import *
np.random.seed(random_seed)

print("=========================training PredNet on JAAD=====================")  #

# PredNet Model parameters (Recommend NOT to change)
n_channels, im_height, im_width = (3, 256, 456)  #changed
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
#nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
parallel_model = multi_gpu_model(model, gpus=num_gpus) #multi-gpu model -- modified
parallel_model.compile(loss='mean_absolute_error', optimizer='adam')
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    #if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True, save_weights_only=True))

history = parallel_model.fit_generator(imageLoader(fileList_train, batch_size=1), len(fileList_train)//batch_size, nb_epoch,callbacks=callbacks,
                    validation_data=imageLoader(fileList_val, batch_size=1), validation_steps=len(fileList_val)//batch_size)


print(history.history)

if save_model:
    json_string = model.to_json()
    with open(json_file_name, "w") as f:
        f.write(json_string)
    model.save_weights(weights_file)
