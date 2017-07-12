from __future__ import print_function
import tensorflow as tf
import numpy as np
#from scipy.misc import imread, imresize
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle

import pickle 
from tflearn.data_utils import image_preloader
import h5py
import math

import random
import time








def random_flip_right_to_left(image_batch):
    result = []
    #print("Flipping images")
    #print(len(image_batch))
    for n in range(len(image_batch)):
        #print(image_batch[n].shape)
        if bool(random.getrandbits(1)):
            result.append(image_batch[n][:,::-1,:])
        else:
            result.append(image_batch[n])
    return result

def random_crop(image_batch):
    result = []
    #print("Cropping images")
    for n in range(image_batch.shape[0]):
        #print(image_batch[n].shape)
        start_x = random.randint(0,39)
        start_y = random.randint(0,39)
        result.append(image_batch[n][start_y:start_y+448,start_x:start_x+448,:])
        #print(result[n].shape)
    return result


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []     ## Parameters in this list will be optimized when only last layer is being trained 
        self.parameters = []                ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()                   ## Create Convolutional layers
        self.fc_layers()                    ## Create Fully connected layer
        self.weight_file = weights          
        #self.load_weights(weights, sess)


    def convlayers(self):
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
            print('Adding Data Augmentation')
            #self.parameters = []
            #self.last_layer_parameters = []
            #images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),images)     ## Data augmentation


        # conv1_1
        with tf.variable_scope("conv1_1"):
            weights = tf.get_variable("W", [3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv1_2
        with tf.variable_scope("conv1_2"):
            weights = tf.get_variable("W", [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv1_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope("conv2_1"):
            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]



        # conv2_2
        with tf.variable_scope("conv2_2"):
            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv2_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope("conv3_1"):
            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv3_2
        with tf.variable_scope("conv3_2"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv3_3
        with tf.variable_scope("conv3_3"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope("conv4_1"):
            weights = tf.get_variable("W", [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool3, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_2
        with tf.variable_scope("conv4_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_3
        with tf.variable_scope("conv4_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope("conv5_1"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool4, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv5_2
        with tf.variable_scope("conv5_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
            

        # conv5_3
        with tf.variable_scope("conv5_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_3 = tf.nn.relu(conv + biases)

            self.parameters += [weights, biases]
            self.special_parameters = [weights,biases]
        '''# Variables created here will be named "conv1/weights", "conv1/biases".
            self.conv5_3 = conv_relu(self.conv5_2, [3, 3, 512, 512], [512])

        print('Shape of conv5_3', self.conv5_3.get_shape())'''

        #self.phi_I = tf.einsum('ijkm,ijkn->imn',self.conv5_3,self.conv5_3)
        self.conv5_3 = tf.transpose(self.conv5_3, perm=[0,3,1,2])

        self.conv5_3 = tf.reshape(self.conv5_3,[-1,512,784])

        
        
        conv5_3_T = tf.transpose(self.conv5_3, perm=[0,2,1])

        self.phi_I = tf.matmul(self.conv5_3, conv5_3_T)

        #self.phi_I = tf.reshape(bcnn,[-1,512*512])
        #print('Shape of phi_I after einsum', self.phi_I.get_shape())

        #self.phi_I = tf.tensordot(self.conv5_3, self.conv5_3, [[1,2],[1,2]])
        self.phi_I = tf.reshape(self.phi_I,[-1,512*512])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I,784.0)  
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I),tf.sqrt(tf.abs(self.phi_I)+1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        print('Shape of z_l2', self.z_l2.get_shape())




    def fc_layers(self):

        with tf.variable_scope('fc-new') as scope:
            fc3w = tf.get_variable('W', [512*512, 100], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            #fc3b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), name='biases', trainable=True)
            fc3b = tf.get_variable("b", [100], initializer=tf.constant_initializer(0.1), trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]


    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come 
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of 
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.weight_file, encoding = 'bytes')
        #print(weights_dict.keys())
        
        # Loop over all layer names stored in the weights dict
        vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
        for op_name in vgg_layers:

            with tf.variable_scope(op_name, reuse = True):
                
              # Loop over list of weights/biases and assign them to their corresponding tf variable
                # Biases
              
              var = tf.get_variable('b', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_b']))
                  
            # Weights
              var = tf.get_variable('W', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_W']))

        with tf.variable_scope('fc-new', reuse = True):
            #last_layer_weights = np.load('last_layers_epoch_10.npz')
            #print('Last layer weights: last_layers_epoch_10.npz')
            last_layer_weights = np.load('last_layers_epoch_20_crop.npz')
            print('Last layer weights: last_layers_epoch_20_crop.npz')
            
            var = tf.get_variable('W', trainable = True)
            print('Adding weights to',var.name)
            session.run(var.assign(last_layer_weights['arr_0'][0]))
            var = tf.get_variable('b', trainable = True)
            print('Adding weights to',var.name)
            session.run(var.assign(last_layer_weights['arr_0'][1]))



if __name__ == '__main__':

    #with tf.device('/cpu:0'):
    train_data = h5py.File('../new_train_488.h5', 'r')
    val_data = h5py.File('../new_val.h5', 'r')
    

    print('Input data read complete')

    X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']

    print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    X_val, Y_val = shuffle(X_val, Y_val)
    #print Y_train[0]
    print("Device placement on. Creating Session")
    
    
    sess = tf.Session()
    #sess = tf.InteractiveSession()
    #with tf.device('/gpu:0'):
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 100])
    #print 'Creating graph'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    
    
    
    
    
    #with tf.device("/gpu:0"):
    print('VGG network created')
    
    
    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
    
    print([_.name for _ in vgg.parameters])

    
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)
    
    check_op = tf.add_check_numerics_ops()
    
    correct_prediction = tf.equal(tf.argmax(vgg.fc3l,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    
    sess.run(tf.global_variables_initializer())

    #vgg.load_weights(sess)
    vgg.load_initial_weights(sess)
    print([_.name for _ in vgg.parameters])

    
    batch_size = 16

    for v in tf.trainable_variables():
        print("Trainable variables", v)
    print('Starting training')

    lr = 0.001
    finetune_step = -1

    val_batch_size = 10
    total_val_count = len(X_val)
    correct_val_count = 0
    val_loss = 0.0
    total_val_batch = int(total_val_count/val_batch_size)
    for i in range(total_val_batch):
        batch_val_x, batch_val_y = X_val[i*val_batch_size:i*val_batch_size+val_batch_size], Y_val[i*val_batch_size:i*val_batch_size+val_batch_size]
        val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})

        pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y})
        correct_val_count+=pred

    print("##############################")
    print("Validation Loss -->", val_loss)
    print("correct_val_count, total_val_count", correct_val_count, total_val_count)
    print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*total_val_count))
    print("##############################")

    validation_accuracy_buffer = []
    for epoch in range(100):
        avg_cost = 0.
        total_batch = int(6000/batch_size)
        X_train, Y_train = shuffle(X_train, Y_train)
        

        for i in range(total_batch):
            batch_xs, batch_ys = X_train[i*batch_size:i*batch_size+batch_size], Y_train[i*batch_size:i*batch_size+batch_size]
            batch_xs = random_crop(batch_xs)
            batch_xs = random_flip_right_to_left(batch_xs)

        
            start = time.time()
            sess.run([optimizer,check_op], feed_dict={imgs: batch_xs, target: batch_ys})
            if i%20==0:
                print('Full BCNN finetuning, time to run optimizer for batch size 16:',time.time()-start,'seconds')
            

            cost = sess.run(loss, feed_dict={imgs: batch_xs, target: batch_ys})
            
            if i % 20 == 0:
                print ('Learning rate: ', (str(lr)))
                if epoch <= finetune_step:
                    print("Training last layer of BCNN_DD")
                else:
                    print("Fine tuning all BCNN_DD")

                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,"Loss:", str(cost))
                print("Training Accuracy -->", accuracy.eval(feed_dict={imgs: batch_xs, target: batch_ys}, session=sess))
                #print(sess.run(vgg.fc3l, feed_dict={imgs: batch_xs, target: batch_ys}))
                


        val_batch_size = 10
        total_val_count = len(X_val)
        correct_val_count = 0
        val_loss = 0.0
        total_val_batch = int(total_val_count/val_batch_size)
        for i in range(total_val_batch):
            batch_val_x, batch_val_y = X_val[i*val_batch_size:i*val_batch_size+val_batch_size], Y_val[i*val_batch_size:i*val_batch_size+val_batch_size]
            val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})

            pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y})
            correct_val_count+=pred

        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, total_val_count)
        print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*total_val_count))
        print("##############################")
        

        if epoch>40:
            validation_accuracy_buffer.append(100.0*correct_val_count/(1.0*total_val_count))

            ## Check if the validation accuracy has stopped increasing
            if len(validation_accuracy_buffer)>10:
                index_of_max_val_acc = np.argmax(validation_accuracy_buffer)
                if index_of_max_val_acc==0:
                    break
                else:
                    del validation_accuracy_buffer[0]

        

    test_data = h5py.File('../new_test.h5', 'r')
    X_test, Y_test = test_data['X'], test_data['Y']
    total_test_count = len(X_test)
    correct_test_count = 0
    test_batch_size = 10
    total_test_batch = int(total_test_count/test_batch_size)
    for i in range(total_test_batch):
        batch_test_x, batch_test_y = X_test[i*test_batch_size:i*test_batch_size+test_batch_size], Y_test[i*test_batch_size:i*test_batch_size+test_batch_size]
        
        pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_test_x, target: batch_test_y})
        correct_test_count+=pred

    print("##############################")
    print("correct_test_count, total_test_count", correct_test_count, total_test_count)
    print("Test Data Accuracy -->", 100.0*correct_test_count/(1.0*total_test_count))
    print("##############################")



