#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import request, make_response
from datetime import *
import time
import random
import json 
from parseSketch import *

app = Flask(__name__)

def get_unique_file_name():
    nowTime = datetime.now().strftime("%Y%m%d%H%M%S")#生成当前的时间
    randomNum = random.randint(0,100)#生成随机数n,其中0<=n<=100
    if randomNum<=10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/save', methods=['POST'])
def save():
    name = get_unique_file_name()
    x = json.loads(request.values.get('x'))
    y = json.loads(request.values.get('y'))
    with open("./data/save/" + name + ".json",'w',encoding='utf-8') as json_file:
        json.dump({'x':x, 'y':y},json_file,ensure_ascii=False)
    
    response = make_response(jsonify(response="Success"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

@app.route('/search',methods=['POST'])
def search():
    name = get_unique_file_name()
    x = json.loads(request.values.get('x'))
    with open("./data/search/" + name + ".json",'w',encoding='utf-8') as json_file:
        json.dump({'x': x},json_file,ensure_ascii=False)

    x = parse(x)
    x = test_by_sketch(x)
    
    response = make_response(jsonify(response="Success"))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='159.226.172.85', port='5000', debug=True)