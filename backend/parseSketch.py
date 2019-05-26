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
        for i in sketch[xi]:
            w = int(i['width'] * i['scaleX'])
            h = int(i['height'] * i['scaleY'])
            
            for a in range(w):
                for b in range(h):
                    x[layer[i['type']]][int(40 * (i['pos']['x']+a) / 500)][int(40 * (i['pos']['y']+b) / 500)] = 1
    return x

#parse('2019052010104108')
