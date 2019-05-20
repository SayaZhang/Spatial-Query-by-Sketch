import pandas as pd
import numpy as np
import json

dPath = './data/search/'

def load(filename, path=dPath):
    with open(path+filename+'.json','r') as f:
        data = json.load(f)
        return data

def get_layer():
    return load('layer', './info/')

def parse(filename):
    d = load(filename)
    layer = get_layer()
    x = np.zeros((15,40,40))
    
    for xi in d['x']:
        w = int(xi['width'] * xi['scaleX'])
        h = int(xi['height'] * xi['scaleY'])
        
        for i in range(w):
            for j in range(h):
                x[layer[xi['type']]][int(4 * (xi['pos']['x']+i) / 500)][int(4 * (xi['pos']['y']+j) / 500)] = 1
    
    return x

parse('2019052010104108')
