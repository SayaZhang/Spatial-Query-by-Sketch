# coding: utf-8

# In[1]:

import os
import pickle
import random
import math

import mxnet as mx
import numpy as np
import pandas as pd
import cv2

from mxnet import autograd, gluon, init
from mxnet import ndarray as nd
from mxnet.gluon.data import DataLoader, dataset
from mxnet.gluon import nn, loss as gloss, data as gdata

from sklearn import neighbors
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist
from numpy import linalg