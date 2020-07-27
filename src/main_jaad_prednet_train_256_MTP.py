#! /usr/bin/python2


#### Multiple Timestep Prediction, extrapolate prednet predictions
#### Latest Revisions X. Du 2020/01

import hickle as hkl
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from prednet import PredNet
from jaad_settings import *


def imageLoader_pred(files, batch_size):
    L = len(files)
    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            filename = files[batch_start]
            X = hkl.load(filename)
            Y=  X  #self.output_mode == 'prediction':  # output actual pixels
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)


orig_weights_file = weights_file  # where weights have been saved (from main_jaad_prednet_train_256.py)
orig_json_file = json_file_name


# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)

layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)



input_shape = list(orig_model.layers[0].batch_input_shape[1:])
# print(orig_model.layers[0])
# print(orig_model.layers[0].batch_input_shape) ##(None, 30, 256, 456, 3)
# print(orig_model.layers[0].batch_input_shape[1:]) ##
input_shape[0] = nt
print(input_shape)
inputs = Input(input_shape)

predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
# model.compile(loss=extrap_loss, optimizer='adam')
print("load models")
parallel_model = multi_gpu_model(model, gpus=num_gpus)  # updated: multi-gpu model
parallel_model.compile(loss=extrap_loss, optimizer='adam')

####model.compile(loss='mean_absolute_error', optimizer='adam')

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    #if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', save_best_only=True, save_weights_only=True))

history = parallel_model.fit_generator(imageLoader_pred(fileList_train, batch_size=1), len(fileList_train)//batch_size, nb_epoch,callbacks=callbacks,
                    validation_data=imageLoader_pred(fileList_val, batch_size=1), validation_steps=len(fileList_val)//batch_size)
#### imageLoader_pred -- this is changed, following SequenceGenerator() class when self.output_mode == 'prediction'

print(history.history)

if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)
    model.save_weights(extrap_weights_file)


