import pandas as pd
import numpy as np
import json

def load(filename, path):
    with open(path+filename+'.json','r') as f:
        data = json.load(f)
        return data

def get_layer():
    return load('layer', './info/')

def parse(sketch):
    #d = load(filename)
    layer = get_layer()
    x = np.zeros((15,40,40))
    
    for xi in sketch:
        w = int(xi['width'] * xi['scaleX'])
        h = int(xi['height'] * xi['scaleY'])
        
        for i in range(w):
            for j in range(h):
                x[layer[xi['type']]][int(40 * (xi['pos']['x']+i) / 500)][int(40 * (xi['pos']['y']+j) / 500)] = 1
    
    return x

#parse('2019052010104108')
