'''
This code is used to extract weights of VGG16 pretrained model.
Weights of all layers are stored in a dictionary.
These weights are used during training the Bilinear CNN.
'''


import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
from tflearn.data_utils import shuffle
import numpy as np
import pickle 
from tflearn.data_utils import image_preloader

def vgg16_base(input):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1',trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    #x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8') 
    x = tflearn.fully_connected(x, 100, activation='softmax', scope='fc8', restore=False)
    return x





num_classes = 100 # num of your dataset

# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
                                     per_channel=True)

# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16_base(x)

sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=500)
regression = tflearn.regression(softmax, optimizer=sgd,
                                loss='categorical_crossentropy')

model = tflearn.DNN(regression, checkpoint_path='vgg_dummy',
                    best_checkpoint_path='vgg_dummy',max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")


model.load("/home/adoke/tf_tutorial/aircrafts_new/new_train_val/vgg16.tflearn", weights_only=True)


vgg_weights_dict = {}
vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7']

for layer_name in vgg_layers:
  print layer_name
  base_layer = tflearn.variables.get_layer_variables_by_name(layer_name)
  vgg_weights_dict[layer_name] = [model.get_weights(base_layer[0]),model.get_weights(base_layer[1])]
    
pickle.dump( vgg_weights_dict, open( "./vgg_weights.p", "wb" ) )  


