
# coding: utf-8

# In[1]:


import os
import pickle
import random
import math

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


# ## Data Augment

# In[2]:


x_0 = pickle.load(open('output/train/x_0.pkl', 'rb'))


# In[3]:


RANGE = len(x_0[0])
SAMPLENUM = 20
BASE = 10


# In[ ]:


indices = np.load('output/train/indices.npy')


# In[ ]:


x_1 = []
for i in range(1,SAMPLENUM+1):
    x_1.append(pickle.load(open('output/train/x_'+str(i)+'.pkl', 'rb')))


# In[ ]:


X1, X2, X3, Y = [], [], [], []
for i in range(RANGE):
    for j in range(SAMPLENUM):
        X1.append(x_0[0][i])
        X2.append(x_1[j][i])
        X3.append(x_1[np.random.randint(0, SAMPLENUM)][int(indices[i][BASE+j])])
        Y.append(x_0[1][i])


# In[ ]:


X1 = np.array(X1).astype(np.float32)
X2 = np.array(X2).astype(np.float32)
X3 = np.array(X3).astype(np.float32)    
Y = np.array(Y).astype(np.float32)

x = np.array(x_0[0]).astype(np.float32)
y = np.array(x_0[1]).astype(np.float32)


# In[ ]:


test = pickle.load(open('output/test_500_2.pkl', 'rb'))
x_test = np.array(test[0]).astype(np.float32)
y_test = np.array(test[1]).astype(np.float32)


# ## Model

# In[18]:


class ArrayDataset():
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._length = len(args[0])
        self._data = []
        for i, data in enumerate(args):
            assert len(data) == self._length,                 "All arrays must have the same length; array[0] has length %d "                 "while array[%d] has %d." % (self._length, i+1, len(data))
            if isinstance(data, nd.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            self._data.append(data)

    def __getitem__(self, idx):
        if len(Xself._data) == 1:
            return self._data[0][idx]
        else:
            return tuple(data[idx] for data in self._data)

    def __len__(self):
        return self._length


# In[ ]:


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


# In[19]:


def evaluate_net(net, train_iter, test_iter, ctx):
    
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

    #normalizer = preprocessing.Normalizer().fit(X)
    #X = normalizer.transform(X)
    nbrs = neighbors.NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    nbrs_all = neighbors.NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
    
    # Predict for test
    for i, (test, label) in enumerate(test_iter):
        test = test.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        if i == 0:
            X_test = net(test).asnumpy()
            Y_test = label.asnumpy()
        else:
            X_test = np.concatenate((X_test, net(test).asnumpy()))
            Y_test = np.concatenate((Y_test, label.asnumpy()))
    
    # Find Nearest Neighbors for test in train base
    mrr, acc1, acc3, acc5, acc10, length = 0.0, 0.0, 0.0, 0.0, 0.0, len(X_test)
    #X_test = normalizer.transform(X_test)    
    distances, indices = nbrs.kneighbors(X_test)
    distances_all, indices_all = nbrs_all.kneighbors(X_test)
    
    for i in range(len(indices)):
        
        y_true = Y_test[i] # True label for test        
        y_predict = Y[indices[i]] # Predict for test
        y_predict_all = Y[indices_all[i]]
        
        rank = np.argwhere(y_predict_all == y_true)[0][0] + 1
        mrr += 1/rank
        
        #print(y_true, y_predict, indices[i])
        
        if y_true in y_predict[:1]:
            acc1 += 1
        
        if y_true in y_predict[:3]:
            acc3 += 1
        
        if y_true in y_predict[:5]:
            acc5 += 1
        
        if y_true in y_predict:
            acc10 += 1
        
    acc1 /= length
    acc3 /= length
    acc5 /= length
    acc10 /= length
    mrr /= length
    
    return [mrr,acc1,acc3,acc5,acc10]


# In[50]:


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


# In[ ]:


def save(params, filename, strip_prefix=''):
    """Save parameters to file.

    Parameters
    ----------
    filename : str
        Path to parameter file.
    strip_prefix : str, default ''
        Strip prefix from parameter names before saving.
    """
    arg_dict = {}
    for param in params.values():
        weight = param._reduce()
        if not param.name.startswith(strip_prefix):
            raise ValueError(
                "Prefix '%s' is to be striped before saving, but Parameter's "
                "name '%s' does not start with '%s'. "
                "this may be due to your Block shares parameters from other "
                "Blocks or you forgot to use 'with name_scope()' when creating "
                "child blocks. For more info on naming, please see "
                "http://mxnet.incubator.apache.org/tutorials/basic/naming.html"%(
                    strip_prefix, param.name, strip_prefix))
        arg_dict[param.name[len(strip_prefix):]] = weight
    nd.save(filename, arg_dict)


# In[ ]:


def project(x, y, X1, X2, X3, Y, x_test, y_test):
    
    ctx = mx.gpu()
    batch_size = 512
    random.seed(47)
    
    ArrayDataset(X1, X2, X3, Y)
    train_iter = gdata.DataLoader(gdata.ArrayDataset(X1, X2, X3, Y), batch_size, shuffle=True)    
    test_iter = gdata.DataLoader(gdata.ArrayDataset(x_test, y_test), batch_size, shuffle=True)
    x_iter = gdata.DataLoader(gdata.ArrayDataset(x, y), batch_size, shuffle=False)
    print('Data Load Success!')
    
#     net = DenseNet()
    
#     conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
#     net = vgg(conv_arch)
    net = SPP_CNN()
#     net = baseNet() 
    #net.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx) 
    
    net.initialize(ctx=ctx, init=init.Xavier())
    print('Build Net Success!')

    triplet_loss = gloss.TripletLoss()
    trainer_triplet = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
    
    print('---------------->')
    best_mrr = 0
    for epoch in range(50):      
        curr_loss = 0.0
        for i, (a, b, c, label) in enumerate(train_iter):            
            a = a.as_in_context(ctx)
            b = b.as_in_context(ctx)
            c = c.as_in_context(ctx)            
            
            anc_ins, pos_ins, neg_ins = a, b, c
            with autograd.record():
                inter1 = net(anc_ins)
                inter2 = net(pos_ins)
                inter3 = net(neg_ins)
                loss = triplet_loss(inter1, inter2, inter3)  # Triplet Loss
            
            loss.backward()
            trainer_triplet.step(batch_size)
            curr_loss = mx.nd.mean(loss).asscalar()
            # print('Epoch: %s, Batch: %s, Triplet Loss: %s' % (epoch, i, curr_loss))
        
        print('Epoch: %s, Triplet Loss: %s' % (epoch, curr_loss)) 
        evas = evaluate_net(net, x_iter, test_iter, ctx) #mrr,acc1,acc3,acc5,acc10
        print(evas[0], ' |', evas[1],' |', evas[2],' |', evas[3], ' |', evas[4])
        
        if evas[0] > best_mrr:
            best_mrr = evas[0]
            save(net.collect_params(), 'SPP+CNN.params')
            #net.export("net", epoch=2)


# In[ ]:


project(x, y, X1, X2, X3, Y, x_test, y_test) #0.3 3 11 7 5

