""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

from collections import namedtuple
import itertools
import glob
import logging
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import windows as win
import salt_baseline as sb
import salt_data as sd

# import matplotlib.pyplot as plt
import pdb

# Training Parameters
learning_rate = 0.0001
batch_size = 32
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])

def scat2d(x, win_params, layer_params):
    """Single layer of 2d scattering
    Args:
        x is input with dim (batch, height, width, 1)
        win_params.filters is complex with dim ......... (depth, height, width, channels)
    """
    real1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def scat2d_to_2d_2layer(x, reuse=tf.AUTO_REUSE, bs=batch_size):
    """
    Args:
        x: in (batch, h, w, 1) shape
    Returns
        (batch, h, w, channels)
    """
    psis = [None,None]
    layer_params = [None,None,None]
    with tf.variable_scope('scat2d_to_2d_2layer', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs

        psis[0] = win.fst2d_psi_factory([7, 7], include_avg=False)
        layer_params[0] = layerO((1,1), 'valid')

        # 107, 107
        U1 = scat2d(x, psis[0], layer_params[0])

        psis[1] = win.fst2d_psi_factory([7, 7], include_avg=False)
        layer_params[1] = layerO((1,1), 'valid')

        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            increasing_psi = win.fst2d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(scat2d(U1[:,:,:,res_i:(res_i+1)], increasing_psi, layer_params[1]))
        

        # 101, 101
        U2 = tf.concat(U2s, 3)
        # swap to (batch, chanU2, h, w)
        U2 = tf.transpose(U2, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U2os = U2.get_shape()
        U2 = tf.reshape(U2, (bs*U2.get_shape()[1], U2.get_shape()[2],U2.get_shape()[3],1))

        # swap to (batch, chanU1, h, w)
        U1 = tf.transpose(U1, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U1os = U1.get_shape()
        U1 = tf.reshape(U1, (bs*U1.get_shape()[1], U1.get_shape()[2], U1.get_shape()[3], 1))

        # now lo-pass

        # each layer lo-passed differently so that (h,w) align bc we
        # want to be able to do 2d convolutions afterwards again
        layer_params[2] = layerO((1,1), 'valid')
        phi = win.fst2d_phi_factory([5,5])

        # filter and separate by original batch via old shape
        S0 = scat2d(x[:,6:-6, 6:-6, :], phi, layer_params[2])
        S0 = tf.reshape(S0, (bs, 1, S0.get_shape()[1], S0.get_shape()[2]))
        S1 = scat2d(U1[:,3:-3,3:-3,:], phi, layer_params[2])
        S1 = tf.reshape(S1, (bs, U1os[1], S1.get_shape()[1],S1.get_shape()[2]))
        S2 = scat2d(U2, phi, layer_params[2])
        S2 = tf.reshape(S2, (bs, U2os[1], S2.get_shape()[1], S2.get_shape()[2]))

        # (batch, chan, h,w)
        feat2d = tf.concat([S0,S1,S2], 1)

    return tf.transpose(feat2d, [0,2,3,1])

# Create the neural network
def pixel_net(x_dict, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']
        # x = tf.reshape(x, shape=[-1, 101, 101, 1])

        # (batch, h, w, chan)
        feat = scat2d_to_2d_2layer(x, reuse)
        fs = feat.get_shape()
        feat = tf.reshape(feat, (fs[0]*fs[1]*fs[2], fs[3]))

        fc1 = tf.layers.dense(feat, 256)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)

        fc3 = tf.layers.dense(fc2, 64)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_training)

        fc4 = tf.layers.dense(fc3, 32, activation=tf.nn.relu)
        out = tf.layers.dense(fc4, 2)

    return out


def kaggle_metric(labels, predictions):
    """
    A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object.
    """

    threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    bs = batch_size
    labels = tf.reshape(labels, [bs, 101**2])
    predictions = tf.cast(tf.reshape(predictions, [bs, 101**2]), dtype=tf.int32)
    
    intx = tf.reduce_sum(tf.multiply(labels, predictions), axis=1)
    union = tf.reduce_sum(tf.add(labels, predictions), axis=1)
    iou = tf.divide(intx, tf.add(union,1))
    mask_present_gt = tf.minimum(tf.reduce_sum(labels, axis=1),1)
    # these metrics methods return a tuple container
    TP = tf.metrics.true_positives_at_thresholds(mask_present_gt, iou, threshes)[0] #  IoU above the threshold.
    FP = tf.metrics.false_positives_at_thresholds(mask_present_gt, iou, threshes)[0] # predicted something, no gt.
    FN = tf.metrics.false_negatives_at_thresholds(mask_present_gt, iou, threshes)[0] # gt but no prediction
    precisions = tf.divide(TP, tf.add(tf.add(TP,tf.add(FP, FN)),1))
    avg_precision = tf.reduce_sum(precisions) / 10
    return [precisions, TP, FP, FN, avg_precision]

def summarize_metrics(metric_op, name, threshes):
    """https://stackoverflow.com/a/50862530/2256243
    """
    shape = metric_op.shape.as_list()
    if shape:  # this is a metric created with any of tf.metrics.*_at_thresholds
        summary_components = tf.split(metric_op, shape[0])
        for i, summary_component in enumerate(summary_components):
            tf.summary.scalar(
                name='{op_name}_{i}'.format(op_name=name, i=threshes[i]),
                tensor=tf.squeeze(summary_component, axis=[0])
            )
    else:  # this already is a scalar metric operator
        tf.summary.scalar(name=summary_components.name, tensor=metric_op)


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = pixel_net(features, dropout, reuse=False,
                            is_training=True)
    logits_test = pixel_net(features, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    notflat_pred_classes = tf.reshape(pred_classes, [batch_size, 101**2])

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'mask': notflat_pred_classes
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss and optimizer
    flat_labels = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=flat_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=flat_labels, predictions=pred_classes)
    iou_op = tf.metrics.mean_iou(labels=flat_labels, predictions=pred_classes, num_classes=2)

    # kaggle spec
    threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    labels = tf.cast(labels, dtype=tf.float32)
    notflat_pred_classes = tf.cast(notflat_pred_classes, dtype=tf.float32)
    intx = tf.reduce_sum(tf.multiply(labels, notflat_pred_classes), axis=1)
    union = tf.reduce_sum(tf.add(labels, notflat_pred_classes), axis=1)
    iou = tf.divide(intx, tf.add(union,1))
    mask_present_gt = tf.minimum(tf.reduce_sum(labels, axis=1),1)

    TP, TP_op = tf.metrics.true_positives_at_thresholds(mask_present_gt, iou, threshes) #  IoU above the threshold.
    FP, FP_op = tf.metrics.false_positives_at_thresholds(mask_present_gt, iou, threshes) # predicted something, no gt.
    FN, FN_op = tf.metrics.false_negatives_at_thresholds(mask_present_gt, iou, threshes) # gt but no prediction
    prec, prec_op = tf.metrics.precision_at_thresholds(mask_present_gt, iou, threshes)

    summarize_metrics(TP_op, 'TP', threshes)
    summarize_metrics(FP_op, 'FP', threshes)
    summarize_metrics(FN_op, 'FN', threshes)
    summarize_metrics(prec_op, 'prec', threshes)

    myevalops = {'2pxwise_accuracy': acc_op,
    '2pxwise_iou': iou_op}

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=myevalops)

    return estim_specs

from window_plot import ScrollThruPlot
import scipy.io as sio
DATA_PATH = os.environ['DATA_PATH']
DATASET_PATH = os.environ['DATASET_PATH']
from window_plot import pyplot_cube

def scat2d_eg():

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    data = mat_contents['indian_pines_corrected'].astype(np.float32)
    data /= np.max(np.abs(data))
    pdb.set_trace()
    egbatch = np.expand_dims(data[:,:,108:109], 0)


    x = tf.placeholder(tf.float32, shape=egbatch.shape)
    feat = scat2d_to_2d_2layer(x, bs=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {x: egbatch}
    data = sess.run(feat, feed_dict)[0]
    

    pyplot_cube(data[:,-1,:], data[:,:,108], np.rot90(data[0,:,:]), title='title',resample_factor=1)

    files = glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/train/images/*.png')
    # now lets look at them
    X = data
    fig, ax = plt.subplots(1, 1)
    tracker = ScrollThruPlot(ax, X, fig)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    pdb.set_trace()

def pixel_eg():
    x = tf.placeholder(tf.float32, shape=(8,117,117,1))
    feat = pixel_net({'images': x}, dropout, reuse=False, is_training=True)

    egbatch = np.random.rand(8,117,117,1)
    # egbatch = egbatch[:8,:,:,:]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {x: egbatch}
    myres = sess.run(feat, feed_dict)

def get_salt_images(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32) / 255.0
        npim_padded = np.pad(npim, ((8,8),(8,8)), 'reflect')
        image_list.append(npim_padded)
        im.close()
    image_list = np.array(image_list)
    return np.expand_dims(image_list, -1)

def get_salt_labels(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/masks/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=int) / 255
        image_list.append(npim.reshape(101**2))
        im.close()
    return np.array(image_list)

def main():

    trainX = get_salt_images(folder='mytrain')
    trainY = get_salt_labels(folder='mytrain')

    valX = get_salt_images(folder='myval')
    valY = get_salt_labels(folder='myval')
    
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binarypix1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': trainX[:3584,:,:,:]}, y=trainY[:3584,:],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        # tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 25 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]}, y=valY[:384,:],
                batch_size=batch_size, shuffle=False)
            # Use the Estimator 'evaluate' method
            e = model.evaluate(input_fn)

            # print("Testing Accuracy:", e['accuracy'])

def eval_masks(outpath='/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/predictions/myval/'):
    valX = get_salt_images(folder='myval')
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/myval/images/*.png'))
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binarypix1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]},
                batch_size=batch_size, shuffle=False)
    gen = model.predict(input_fn)

    for file_i, prediction in enumerate(tqdm(gen, total=384)):
        p_label = prediction['mask']
        pred = np.array(p_label).reshape((101,101))
        plt.imsave(outpath+fileids[file_i], pred, cmap=cm.gray)

    # now get the tail
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': valX[-32:,:,:,:]},
        batch_size=batch_size, shuffle=False)
    gen = model.predict(input_fn)
    for file_i, prediction in enumerate(gen):
        fileid = fileids[404-32+file_i]
        
        p_label = prediction['mask']
        pred = np.array(p_label).reshape((101,101))
        plt.imsave(outpath+fileid, pred, cmap=cm.gray)
        
def kaggle_summary(outpath='/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/predictions/myval/'):
    valX = get_salt_images(folder='myval')
    valY = get_salt_labels(folder='myval')
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/myval/images/*.png'))
    
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binarypix1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]},
                batch_size=batch_size, shuffle=False)
    gen = model.predict(input_fn)

    id_to_pred = np.load('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary13b/val_bin_pred.npy').tolist()
    threshes = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tps = np.zeros(threshes.shape[0])
    fps = np.zeros(threshes.shape[0])
    fns = np.zeros(threshes.shape[0])
    def tp_fp_fn_calc(gt, predicted, tps, fps, fns):
        component1 = np.array(predicted).astype(bool)
        component2 = gt.astype(bool)

        overlap = component1*component2 # Logical AND
        union = component1 + component2 # Logical OR

        iou = overlap.sum()/float(union.sum() + 1e-5)

        mask_present_gt = np.any(gt > 0)
        # these metrics methods return a tuple container
        if mask_present_gt:
            if np.all(predicted < 1):
                fns += np.ones(threshes.shape[0])
            else:
                tps += (threshes < iou).astype(int)
        else:
            if np.any(predicted > 0):
                fps += np.ones(threshes.shape[0])


    for file_i, prediction in enumerate(tqdm(gen, total=384)):
        fileid, file_extension = os.path.splitext(fileids[file_i])

        p_label = prediction['mask']
        p_label = np.array(p_label) * id_to_pred[fileid]
        tp_fp_fn_calc(valY[file_i,:], p_label, tps, fps, fns)

    # now get the tail
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': valX[-32:,:,:,:]},
        batch_size=batch_size, shuffle=False)
    gen = model.predict(input_fn)
    for file_i, prediction in enumerate(gen):
        idx = 404-32+file_i
        fileid, file_extension = os.path.splitext(fileids[idx])
        
        p_label = prediction['mask']
        p_label = np.array(p_label) * id_to_pred[fileid]
        tp_fp_fn_calc(valY[idx,:], p_label, tps, fps, fns)


    precisions = tps / (tps+fps+fns)
    avg_precision = precisions.sum() / threshes.shape[0]
    
    print('%d masks in dataset' % np.sum(valY.sum(axis=1) > 0))
    for idx, thresh in enumerate(threshes):
        print('tp at %f: %f' % (thresh, tps[idx]))
        print('fp at %f: %f' % (thresh, fps[idx]))
        print('fn at %f: %f' % (thresh, fns[idx]))
        print('precisions at %f: %f' % (thresh, precisions[idx]))
    print('avg precision: %f' % (avg_precision))



def kaggle_test(outpath='/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/predictions/'):
    testX = get_salt_images(folder='test')
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/test/images/*.png'))

    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binarypix1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': testX[:17984,:,:,:]},
        batch_size=batch_size, shuffle=False)
    gen = model.predict(input_fn)

    id_to_pred = np.load('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary1/test_bin_pred.npy').tolist()

    with open(outpath+'binarypix2.csv','a') as fd:
        fd.write('id,rle_mask\n')
        for file_i, prediction in enumerate(tqdm(gen, total=17984)):
            fileid, file_extension = os.path.splitext(fileids[file_i])
            
            p_label = prediction['mask']
            pred = np.array(p_label).reshape((101,101)).transpose().reshape(101**2) * id_to_pred[fileid]
            fd.write('%s,%s\n' % (fileid, myrlestring(pred)))

        # now get the tail
        input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': testX[-32:,:,:,:]},
            batch_size=batch_size, shuffle=False)
        gen = model.predict(input_fn)
        for file_i, prediction in enumerate(gen):
            if file_i >= 16:
                fileid, file_extension = os.path.splitext(fileids[18000-32+file_i])
            
                p_label = prediction['mask']
                pred = np.array(p_label).reshape((101,101)).transpose().reshape(101**2) * id_to_pred[fileid]
                fd.write('%s,%s\n' % (fileid, myrlestring(pred)))



if __name__ == '__main__':
    kaggle_summary()

    # lets look at the result images with the scroll thru vis
    # then do the mnist like network on binary and see results (with PCA layer in between)
    # and research what they do for semantic segmentation, u net like stuff

    # later concatenate in 2d wavelet features
