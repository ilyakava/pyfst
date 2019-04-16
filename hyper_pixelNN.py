"""FST-CNN network
"""
from __future__ import division, print_function, absolute_import

from collections import namedtuple
import h5py
import hdf5storage
import itertools
import glob
import logging
import operator
import os
import random

import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import windows as win
from rle import myrlestring
import salt_baseline as sb
import salt_data as sd
import fst3d_feat as fst

# import matplotlib.pyplot as plt
import pdb

DATA_PATH = '/scratch0/ilya/locDoc/data/hyperspec'
DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'

# Training Parameters
learning_rate = 0.0001
batch_size = 16
num_steps = 300
N_TEST = 2048
#N_TEST = 16
#N_TRAIN = 540
#N_TRAIN = 16
EVAL_PERIOD = 10

# Network Parameters
n_classes=8 # 1-9 for PaviaU and PaviaCR
# dropout = 0.25 # Dropout, probability to drop a unit
dropout = 0.6

layerO = namedtuple('layerO', ['strides', 'padding'])

PaviaU_to_PaviaCR_lab = {
    9: 9,
    8: 4,
    7: 7,
    6: 5,
    5: 0,
    4: 2,
    3: 0,
    2: 3,
    1: 6
}
PaviaCR_to_PaviaU_lab = {
    9: 9,
    8: 0,
    7: 7,
    6: 1,
    5: 6,
    4: 8,
    3: 2,
    2: 4,
    1: 0
}

def scat3d_to_3d_3x3_2layer(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None):
    """Computes features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
    Output:
        center pixel feature vector
    """
    assert len(layer_params) == 3, 'this network is 2 layers only'
    assert len(psis) == 2, 'this network is 2 layers only'

    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)
        # x is (1, bands, h, w, 1)

        U1 = fst.scat3d(x, psis[0], layer_params[0])
        # U1 is (1, bands, h, w, lambda1)

        U1 = tf.transpose(U1, [0, 4, 1, 2, 3])
        # U1 is (1, lambda1, bands, h, w)

        lambda1_d = 4; band1_d = 3
        lambda2_d = 4; band2_d = 4
        lambdax_d = 5;
        U1 = tf.layers.max_pooling3d(U1, (lambda1_d,band1_d,1), (lambda1_d,band1_d,1), padding='same')

        U1 = tf.transpose(U1, [1,2,3,4,0])
        # U1 is (lambda1, bands, h, w, 1)
        
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params[::lambda1_d]):
            increasing_psi = win.fst3d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(fst.scat3d(U1[res_i:(res_i+1),:,:,:,:], increasing_psi, layer_params[1]))
        
        U2 = tf.concat(U2s, 4)
        # U2 is (1,bands,h,w,lambda2)
        
        U2 = tf.transpose(U2, [0, 4, 1, 2, 3])
        # U2 is (1, lambda2, bands, h, w)
      
        U2 = tf.layers.max_pooling3d(U2, (lambda2_d,band2_d,1), (lambda2_d,band2_d,1), padding='same')
        
        U2 = tf.transpose(U2, [1, 2, 3, 4, 0])
        # U2 is (lambda2, bands, h, w, 1)

        # convolve with phis
        S2 = fst.scat3d(U2, phi, layer_params[2])

        def slice_idxs(sig_size, kernel_size):
            def slice_idx(s, k, f):
                if k % 2 == 0:
                    raise('not implemented even padding')
                else:
                    return int((s - k - f)//2)
            final_size = [1,3,3]
            return [slice_idx(s,k,f-1) for s,k,f in zip(sig_size, kernel_size,final_size)]

        [p1b, p1h, p1w] = slice_idxs(U1.shape[1:4], psis[1].kernel_size)
        [p2b, p2h, p2w] = slice_idxs(x.shape[1:4], psis[0].kernel_size)

        S1 = fst.scat3d(U1[:, :,(p1h):-(p1h), (p1w):-(p1w), :], phi, layer_params[2])
        S0 = fst.scat3d(x[:, :,(p2h):-(p2h), (p2w):-(p2w), :], phi, layer_params[2])

        # just to get the size down to 1
        S0 = tf.reshape(S0, [-1, 3, 3, 1])
        S1 = tf.reshape(S1, [-1, 3, 3, 1])
        S2 = tf.reshape(S2, [-1, 3, 3, 1])
        
        SX = tf.concat([S0,S1,S2], 0)
        # SX is (lambdax, 3, 3, 1)
        
        SX = tf.expand_dims(SX, 0)
        SX = tf.layers.max_pooling3d(SX, (lambdax_d,1,1), (lambdax_d,1,1), padding='same')

        SX = tf.squeeze(SX)
        # SX is (channels, h, w)

    return SX

# Create the neural network
def hyper_3x3_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']

        # 1x1 conv replaces PCA step
        # x should be (batch, channel, h, w)
        conv1 = tf.layers.conv2d(x, 1024, 1, data_format='channels_first')
        # Convolution Layer with filters of size 3
        conv2 = tf.layers.conv2d(conv1, 512, 3, activation=tf.nn.relu, padding='same', data_format='channels_first')
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2, data_format='channels_first')
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 500)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        fc2 = tf.layers.dense(fc1, 100)
        out = tf.layers.dense(fc2, n_classes)

    return out

def deng_cnn(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']
        # x should be (batch, h, w, channel)

        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(x, 64, 4, activation=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(conv1, 4, 1, padding='same')

        fc1 = tf.contrib.layers.flatten(conv1)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)

    return out

def yu_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']
        # x should be (batch, h, w, channel)

        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(x, 128, 1, activation=None)
        conv1 = tf.layers.average_pooling2d(conv1, 2, 1, padding='same')
        conv1 = tf.layers.dropout(conv1, rate=dropout, training=is_training)
        # conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, 64, 1, activation=None)
        conv2 = tf.layers.average_pooling2d(conv2, 2, 1, padding='same')
        conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)

        conv3 = tf.layers.conv2d(conv2, n_classes, 1, activation=None)
        out = tf.layers.average_pooling2d(conv3, 5, 1)

    return tf.squeeze(out)

def yu2_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']
        # x should be (batch, h, w, channel)

        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(x, 128, 1, activation=None)
        conv1 = tf.layers.average_pooling2d(conv1, 2, 1, padding='same')
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, 64, 1, activation=None)
        conv2 = tf.layers.average_pooling2d(conv2, 2, 1, padding='same')
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, n_classes, 1, activation=None)
        out = tf.layers.average_pooling2d(conv3, 5, 1)

    return tf.squeeze(out)

def hyper_3x3_yunet(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']
        x = tf.transpose(x, [0, 2, 3, 1])

        # 1x1 conv replaces PCA step
        # x should be (batch, channel, h, w)
#         conv1 = tf.layers.conv2d(x, 4096, 1, activation=None)
#         conv1 = tf.layers.average_pooling2d(conv1, 2, 1, padding='same')
#         conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(x, 128, 1, activation=None)
    
        conv2 = tf.layers.batch_normalization(conv2)
        #conv2 = tf.layers.average_pooling2d(conv2, 2, 1, padding='same')
        #conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)

#         conv3 = tf.layers.conv2d(conv2, 256, 1, activation=None)
#         conv3 = tf.layers.average_pooling2d(conv3, 2, 1, padding='same')
#         conv3 = tf.nn.relu(conv3)
        
        conv4 = tf.layers.conv2d(conv2, 64, 1, activation=None)
        conv4 = tf.layers.average_pooling2d(conv4, 2, 1, padding='same')
        conv4 = tf.layers.dropout(conv4, rate=dropout, training=is_training)

        conv5 = tf.layers.conv2d(conv4, n_classes, 1, activation=None)
        out = tf.layers.average_pooling2d(conv5, 3, 1)

    return tf.squeeze(out)

# network=hyper_3x3_net
network=hyper_3x3_yunet
#network=deng_cnn

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    

    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = network(features, dropout, reuse=False,
                            is_training=True, n_classes=n_classes)
    logits_test = network(features, dropout, reuse=True,
                            is_training=False, n_classes=n_classes)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # tf.summary.scalar('min', loss_op)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# traintestfilename = 'PaviaU_gt_traintest_s60_1_dd069a.mat'
# traintestfilename = 'PaviaU_gt_traintest_coarse_128px128p.mat'
#traintestfilename = 'PaviaU_gt_traintest_coarse_32px32p.mat'

# masks
trainfilename = 'PaviaU_gt_traintest_coarse_32px32p.mat'
testfilename = trainfilename
# 2 masks are not currently supported, if datasets for train/test are different
# then whole image is used

trainimgname, trainimgfield = 'PaviaU.mat', 'paviaU'
testimgname, testimgfield
trainlabelname, trainlabelfield = 'PaviaU_gt.mat', 'paviaU_gt'
testlabelname, testlabelfield = 'PaviaU_gt.mat', 'paviaU_gt'

# masks for train/test sets
mat_contents = None
try:
    mat_contents = sio.loadmat(os.path.join(DATA_PATH, trainfilename))
except:
    mat_contents = hdf5storage.loadmat(os.path.join(DATA_PATH, trainfilename))
train_mask = mat_contents['train_mask'].astype(int).squeeze()
try:
    mat_contents = sio.loadmat(os.path.join(DATA_PATH, testfilename))
except:
    mat_contents = hdf5storage.loadmat(os.path.join(DATA_PATH, testfilename))
test_mask = mat_contents['test_mask'].astype(int).squeeze()
# labels
mat_contents = sio.loadmat(os.path.join(DATASET_PATH, testlabelname))
test_labels = mat_contents[testlabelfield]
mat_contents = sio.loadmat(os.path.join(DATASET_PATH, trainlabelname))
train_labels = mat_contents[trainlabelfield]
# spectra
mat_contents = sio.loadmat(os.path.join(DATASET_PATH, trainimgname))
traindata = mat_contents[trainimgfield].astype(np.float32)
traindata /= np.max(np.abs(traindata))
mat_contents = sio.loadmat(os.path.join(DATASET_PATH, testimgname))
testdata = mat_contents[testimgfield].astype(np.float32)
testdata /= np.max(np.abs(testdata))
data = traindata

def get_train_test_data_raw(addl_padding=(4,4,0)):
    reuse = tf.AUTO_REUSE

    

    [height, width, nbands] = data.shape

    all_pixels = np.array(list(itertools.product(range(width),range(height))))

    ap = np.array(addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), 'wrap')
    
    # modify the train labels to be like the test labels
    CR2U = np.vectorize(lambda l: PaviaCR_to_PaviaU_lab[l])
    U2CR = np.vectorize(lambda l: PaviaU_to_PaviaCR_lab[l])
    if trainfilename != testfilename:
        if 'PaviaU' in testfilename: 
            train_labels = CR2U(test_labels)
        elif 'Pavia_center' in testfilename:
            train_labels = U2CR(test_labels)
        train_pixels = np.array(list(itertools.product(range(traindata.shape[1]),range(traindata.shape[0]))))
        test_pixels = np.array(list(itertools.product(range(testdata.shape[1]),range(testdata.shape[0]))))
    else:
        train_pixels = np.array(filter(lambda (x,y): train_labels[y,x]*train_mask[x*height+y] != 0, all_pixels))
        test_pixels = np.array(filter(lambda (x,y): test_labels[y,x]*test_mask[x*height+y] != 0, all_pixels))
    

    train_pixels_list = train_pixels.tolist()
    random.shuffle(train_pixels_list)
    train_pixels = np.array(train_pixels_list)
    

    test_pixels_list = test_pixels.tolist()
    random.shuffle(test_pixels_list)
    test_pixels = np.array(test_pixels_list)

    print("Train / Test split is %i / %i" % (train_pixels.shape[0], test_pixels.shape[0]))

    batch_item_shape = tuple(map(operator.add, addl_padding, (1,1,data.shape[2])))

    trainX = np.zeros((train_mask.sum(),) + batch_item_shape, dtype=np.float32)
    trainY = np.zeros((train_mask.sum(),))
    valX = np.zeros((test_mask.sum(),) + batch_item_shape, dtype=np.float32)
    valY = np.zeros((test_mask.sum(),))


    for pixel_i, pixel in enumerate(tqdm(train_pixels[:,:], desc='Loading train data: ')):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        trainX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        trainY[pixel_i] = train_labels[pixel_y,pixel_x] - 1

    for pixel_i, pixel in enumerate(tqdm(test_pixels[:N_TEST,:], desc='Loading test data: ')):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        valX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        valY[pixel_i] = test_labels[pixel_y,pixel_x] - 1

    return trainX, trainY, valX, valY

def get_train_test_data():
    reuse = tf.AUTO_REUSE

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
    data = mat_contents['paviaU'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
    labels = mat_contents['paviaU_gt']

    netO = fst.Pavia_net()
    # from pavia net
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((1,1,1), 'valid')
    psis=[psi,psi]
    layer_params=[layer_params, layer_params, layer_params]
    # addl_padding = (18,18,18)
    addl_padding = (20,20,18)
    

    [height, width, nbands] = data.shape

    all_pixels = np.array(list(itertools.product(range(width),range(height))))

    ap = np.array(addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), 'wrap')
    
    train_pixels = np.array(filter(lambda (x,y): labels[y,x]*train_mask[x*height+y] != 0, all_pixels))
    test_pixels = np.array(filter(lambda (x,y): labels[y,x]*test_mask[x*height+y] != 0, all_pixels))
    
    train_pixels_list = train_pixels.tolist()
    random.shuffle(train_pixels_list)
    train_pixels = np.array(train_pixels_list)
    

    test_pixels_list = test_pixels.tolist()
    random.shuffle(test_pixels_list)
    test_pixels = np.array(test_pixels_list)

    print("Train / Test split is %i / %i" % (train_pixels.shape[0], test_pixels.shape[0]))

    batch_item_shape = tuple(map(operator.add, netO.addl_padding, (3,3,data.shape[2])))

    x = tf.placeholder(tf.float32, shape=batch_item_shape)
    feat = scat3d_to_3d_3x3_2layer(x, reuse, psis, phi, layer_params)
    feat_shape = tuple([int(d) for d in feat.shape])

    trainX = np.zeros((train_mask.sum(),) + feat_shape, dtype=np.float32)
    trainY = np.zeros((train_mask.sum(),))
    valX = np.zeros((test_mask.sum(),) + feat_shape, dtype=np.float32)
    valY = np.zeros((test_mask.sum(),))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(train_pixels[:,:], desc='Loading train data: ')):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
            feed_dict = {x: subimg}
            trainX[pixel_i,:,:,:] = sess.run(feat, feed_dict)
            trainY[pixel_i] = labels[pixel_y,pixel_x] - 1

        for pixel_i, pixel in enumerate(tqdm(test_pixels[:N_TEST,:], desc='Loading test data: ')):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
            feed_dict = {x: subimg}
            valX[pixel_i,:,:,:] = sess.run(feat, feed_dict)
            valY[pixel_i] = labels[pixel_y,pixel_x] - 1

    return trainX, trainY, valX, valY

def main():
#    trainX, trainY, valX, valY = get_train_test_data_raw((4,4,0))
    trainX, trainY, valX, valY = get_train_test_data()
    
    # model_dir = '/scratch0/ilya/locDoc/data/hypernet/models/threexthree'
    model_dir = '/scratch0/ilya/locDoc/data/hypernet/models/test'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    best_acc = 0

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'subimages': trainX[:,:,:,:]}, y=trainY[:],
            batch_size=batch_size, num_epochs=EVAL_PERIOD, shuffle=True)
        # Train the Model
        tf.logging.set_verbosity(tf.logging.ERROR)
        model.train(input_fn, steps=None) # 

        # if i % EVAL_PERIOD == 0:
        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'subimages': valX[:N_TEST,:,:,:]}, y=valY[:N_TEST],
            batch_size=batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        best_acc = max(best_acc, e['accuracy'])
        print("{:06d}: Testing Accuracy: {:.4f} (Best: {:.4f})".format(i, e['accuracy'], best_acc))

if __name__ == '__main__':
    main()
