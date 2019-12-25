# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:08:50 2019

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 05:30:17 2019

@author: ASUS
"""
#pip install flask
#python api.py
#localhost:5000

import flask
from flask import request, jsonify, render_template    
import sqlite3

app = flask.Flask(__name__)
app.config["DEBUG"] = True

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
#E://ntrodemos/grievanceapp/json-file/json file/
with open("intents.json") as file:
    data = json.load(file)
#E://ntrodemos/grievanceapp/json-file/json file/
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])],name='Input_layer')
net = tflearn.fully_connected(net, 8,activation='relu', name='Hidden_layer_1')
net = tflearn.fully_connected(net, 8,activation='relu', name='Hidden_layer_2')
net = tflearn.fully_connected(net, len(output[0]), activation="softmax",name='Output_layer')
net = tflearn.regression(net, optimizer='sgd',learning_rate=2, loss='mean_square')

model = tflearn.DNN(net)
#E://ntrodemos/grievanceapp/
model.load("model.tflearn")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat(inp):
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return(random.choice(responses))

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/message', methods=['POST'])
def message():
    data = request.form    
    messageret= chat(data['message']) 
    print(data['message'])
    return render_template("home.html", messageret=messageret)

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

app.run(debug=True)