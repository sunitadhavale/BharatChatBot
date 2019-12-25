# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:35:39 2019

@author: ASUS
"""
#579b464db66ec23bdd0000019d5b5b18bd014bd040d5422cc8c004c3

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

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #98.51 accuracy
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#E://ntrodemos/grievanceapp/json-file/json file/

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])],name='Input_layer')
net = tflearn.fully_connected(net, 8,activation='relu', name='Hidden_layer_1')
net = tflearn.fully_connected(net, 8,activation='relu', name='Hidden_layer_2')
net = tflearn.fully_connected(net, len(output[0]), activation="softmax",name='Output_layer')
net = tflearn.regression(net, optimizer='sgd',learning_rate=2, loss='mean_square')

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#E://ntrodemos/grievanceapp/
model.save("model.tflearn")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

#run chat() for testing at server side
#chat()

result = model.evaluate(training, output)
print('Accuracy is %0.2f%%' % (result[0] * 100))

#import tensorflow as tf
#import tensorflow.contrib.slim as slim
#
#def model_summary():
#    model_vars = tf.trainable_variables()
#    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
#    
#model_summary()
    
#import matplotlib.pyplot as plt
#
#pred_y = model.predict(training)
#
#plt.plot(output, color = 'red', label = 'Real data')
#plt.plot(pred_y, color = 'blue', label = 'Predicted data')
#plt.title('Prediction')
#plt.legend()
#plt.show()