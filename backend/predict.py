
# coding: utf-8

import os
import pickle
import random
import math
import json

import mxnet as mx
import numpy as np
import pandas as pd

from mxnet import autograd, gluon, init
from mxnet import ndarray as nd
from mxnet.gluon.data import DataLoader, dataset
from mxnet.gluon import nn, loss as gloss, data as gdata

from sklearn import neighbors
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist
from numpy import linalg
from parseSketch import *

#label_info = pd.read_csv('../../train/label_info.csv')

def spatial_pyramid_pool(previous_conv, num_sample, shape, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    ## NCHW
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(H / out_pool_size[i]))
        w_wid = int(math.ceil(W / out_pool_size[i]))
        
        h_pad = int((h_wid*out_pool_size[i] - H + 1) / 2)
        w_pad = int((w_wid*out_pool_size[i] - W + 1) / 2)
        
        maxpool = nn.MaxPool2D(pool_size = (h_wid, w_wid), strides=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        #print(x.shape)
        
        if(i == 0):
            spp = x.reshape((N, -1))
        else:
            spp = nd.concat(
                spp, 
                x.reshape((N, -1)), 
                dim=1
            )
    
    return spp

def predict(net, nbrs, Y, test_iter, ctx): 
    # Predict for test
    for i, (test) in enumerate(test_iter):
        test = test.as_in_context(ctx)
        #label = label.as_in_context(ctx)
        
        if i == 0:
            X_test = net(test).asnumpy()
            #Y_test = label.asnumpy()
        else:
            X_test = np.concatenate((X_test, net(test).asnumpy()))
            #Y_test = np.concatenate((Y_test, label.asnumpy()))
      
    distances, indices = nbrs.kneighbors(X_test)
    return Y[indices[:12]]

class SPP_CNN(nn.Block):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''
    def __init__(self, **kwargs):
        super(SPP_CNN, self).__init__(**kwargs)
        self.output_num = [4, 2, 1]
        
        self.conv1 = nn.Conv2D(channels=96, kernel_size=11, strides=4,activation='relu')
        
        self.conv2 = nn.Conv2D(channels=256, kernel_size=7, padding=2, activation='relu')
        self.BN1 = nn.BatchNorm()

        self.conv3 = nn.Conv2D(channels=384, kernel_size=5, padding=1, activation='relu')
        self.BN2 = nn.BatchNorm()

        #self.conv4 = nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu')
        #self.BN3 = nn.BatchNorm()
        
        #self.conv5 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        #self.BN4 = nn.BatchNorm()
                               
        self.pool1 = nn.MaxPool2D(pool_size=3, strides=2)
        self.pool2 = nn.MaxPool2D(pool_size=3, strides=2)
        self.pool3 = nn.MaxPool2D(pool_size=3, strides=2)
        
        self.drop = nn.Dropout(0.3)
        #self.drop2 = nn.Dropout(0.1)
        
        self.fc1 = nn.Dense(4096)
        self.fc2 = nn.Dense(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.BN1(x)
        x = self.conv3(x)
        x = self.BN2(x)
        #x = self.conv4(x)
        #x = self.BN3(x)
        #x = self.conv5(x)
        #x = self.BN4(x)
        #x = self.pool1(x)

        x = spatial_pyramid_pool(x, 1, x.shape, self.output_num)
        
        fc1 = self.fc1(self.drop(x))
        fc2 = self.fc2(fc1)
        
        return fc2

def get_train_base():

    x_0 = pickle.load(open('../../train/x_0.pkl', 'rb'))

    x = np.array(x_0[0]).astype(np.float32)
    y = np.array(x_0[1]).astype(np.float32)

    ctx = mx.cpu()
    batch_size = 512
    random.seed(47)

    train_iter = gdata.DataLoader(gdata.ArrayDataset(x, y), batch_size, shuffle=False)
        
    net = SPP_CNN()
    net.load_parameters('../model/model.params')
    print('Build Net Success!')
    
    # Predict for train
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        if i == 0:
            X = net(data).asnumpy()
            Y = label.asnumpy()
        else:
            X = np.concatenate((X, net(data).asnumpy()))
            Y = np.concatenate((Y, label.asnumpy()))

    pickle.dump([X,Y], open('../../train/train_base.pkl', 'wb'))

def search(x_test):

    ctx = mx.cpu()
    batch_size = 512
    random.seed(47)
        
    test_iter = gdata.DataLoader(gdata.ArrayDataset(x_test), batch_size, shuffle=True)
    
    net = SPP_CNN()
    net.load_parameters('../model/model.params')
    print('Build Net Success!')
        
    train_base = pickle.load(open('../../train/train_base.pkl', 'rb'))
    nbrs = neighbors.NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_base[0])

    result = predict(net, nbrs, train_base[1], test_iter, ctx)
    return result

def get_sketch(fname):   
    with open("./data/search/"+fname+".json", 'r') as f:
        x = json.load(f)
    x = parse(x)
    return(x)

def test_by_data():
    test = pickle.load(open('../../train/test_500_2.pkl', 'rb'))
    x_test = np.array(test[0]).astype(np.float32)

    return search(x_test)

def test_by_sketch():
    fname = '2019052010104108'
    test = get_sketch(fname)
    x_test = np.array(test).astype(np.float32)
    return search(x_test)

fname = '20190521102020010'
get_sketch(fname)