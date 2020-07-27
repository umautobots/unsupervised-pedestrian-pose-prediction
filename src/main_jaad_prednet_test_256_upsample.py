#! /usr/bin/python2


#### use pre-processed jaad dataset (4,nt=30,256,456,3) files, re-train prednet and test
#### Latest Revision X. Du 2020/01

#! /usr/bin/python2

import hickle as hkl
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from jaad_settings import *


print("=========================saving PredNet frame predicition results (high-res): this process may take a while... ====================")  


X_hat = hkl.load(os.path.join(weights_dir, 'X_hat.hkl'))
X_test = hkl.load(os.path.join(weights_dir, 'X_test.hkl'))

num_seq_test = X_hat.shape[0]
nt = X_hat.shape[1] 
for i in range(num_seq_test):
    print("==Processing X_test #" + str(i) + " out of " + str(num_seq_test) + "==")
    for j in range(nt):
        print("nt= #" + str(j) + " out of " + str(nt) + "==")
        im_small_hat = X_hat[i,j]
        print(im_small_hat.shape)
        plt.imshow(im_small_hat, interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off',
                        labelleft='off')
        plt.savefig(X_hat_images_dir + 'Seq{:05d}_nt{:02d}.png'.format(i,j), bbox_inches='tight',
                    pad_inches=0, dpi=1000)

        plt.clf()

        im_small_test = X_test[i,j]
        print(im_small_test.shape)
        plt.imshow(im_small_test, interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off',
                        labelleft='off')
        plt.savefig(X_test_images_dir + 'Seq{:05d}_nt{:02d}.png'.format(i,j), bbox_inches='tight',
                    pad_inches=0, dpi=1000)
        plt.clf()
        
print("that's all folks!")