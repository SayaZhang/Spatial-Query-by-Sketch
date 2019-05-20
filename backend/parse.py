import pandas as pd
import json

# load json data from file
def load(filename):
    with open('./Data_Collection/'+filename+'.json','r') as f:
        data = json.load(f)
        return data

d = load('2019051514375377')
print(d)
