# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''

import numpy as np
import warnings
import h5py
import os
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils.np_utils import to_categorical
from keras.models import save_model,load_model

import shutil
import pickle

stored_values=pickle.load(open('resnet50.pkl','r'))

import cv2

def prep_image(fname=None):

    im = cv2.imread(fname)
    im = cv2.resize(im,(224,224))
    im = im.transpose(2,0,1).astype(np.float32)
    im = im - stored_values['mean_image']
    return im
    

image_dir = '/home/enuok/zhanqi/data/20160905/'

image_list =os.listdir(image_dir)

image_path = [image_dir+im for im in image_list]

model = load_model('final_model.h5')

f = open('result.txt','w')

for img_path in image_path:
    print img_path

    img = prep_image(img_path)
    img = np.expand_dims(img,0)
    
    output = model.predict(img)
    output = np.squeeze(output)
    output = np.argmax(output)
    print output
    f.write(img_path+','+str(output)+'\n') 
    if output==0:
        shutil.copy(img_path,'./0')
    elif output==1:
        shutil.copy(img_path,'./1')
    else:
        print 'wrong'
