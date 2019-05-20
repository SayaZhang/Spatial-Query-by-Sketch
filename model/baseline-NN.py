# # Baseline-NearestNeighbors

# In[ ]:

import os
import pickle
import random
import math

import numpy as np
import pandas as pd

from sklearn import neighbors
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist
from numpy import linalg

x_0 = pickle.load(open('output/train/x_0.pkl', 'rb'))

x = np.array(x_0[0]).astype(np.float32)
y = np.array(x_0[1]).astype(np.float32)


# In[ ]:


test = pickle.load(open('output/test_500_2.pkl', 'rb'))
x_test = np.array(test[0]).astype(np.float32)
y_test = np.array(test[1]).astype(np.float32)


# In[ ]:


X = []
for i in range(len(x)):
    X.append(x[i].flatten())

X_test = []
for i in range(len(x_test)):
    X_test.append(x_test[i].flatten())

X = np.array(X).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)


# In[ ]:


print(X.shape,y.shape,X_test.shape,y_test.shape)


# In[ ]:


# Find Nearest Neighbors for test in train base
mrr, acc1, acc3, acc5, acc10, length = 0.0, 0.0, 0.0, 0.0, 0.0, len(x_test)

nbrs = neighbors.NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
nbrs_all = neighbors.NearestNeighbors(n_neighbors=len(x), algorithm='ball_tree').fit(X)
   
distances, indices = nbrs.kneighbors(X_test)
distances_all, indices_all = nbrs_all.kneighbors(X_test)
    
for i in range(len(indices)):
        
    y_true = y_test[i] # True label for test        
    y_predict = y[indices[i]] # Predict for test
    y_predict_all = y[indices_all[i]]
        
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
        
    print(i,acc1,acc3,acc5,acc10)
        
acc1 /= length
acc3 /= length
acc5 /= length
acc10 /= length
mrr /= length

print(acc1,acc3,acc5,acc10,mrr)
