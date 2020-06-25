# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:53:40 2020

@author: Supernova
"""
from scipy.special import softmax
import numpy as np

corpus = 'this is my first implementation of skip gram method'
dictionnaire = corpus.split(' ')
word_2_id = {w:i for i,w in enumerate(dictionnaire)} 
id_2_word = {i:w for i,w in enumerate(dictionnaire)}

def one_hot_encoding(word, dic):
    array = np.zeros(len(dic))
    array[word_2_id[word]] = 1
    return array

def data_processin(corpus, windows_size):  
    x_train = list()
    y_train = list()
    
    for i,elm in enumerate(dictionnaire):
        Centre = elm
        x_train.append(one_hot_encoding(Centre, dictionnaire))
        
        for j in range(i-windows_size,i+windows_size + 1):
            y = list()
            if(j != i and j <= len(dictionnaire)-1 and j>= 0):
                y_train.append(one_hot_encoding(dictionnaire[j], dictionnaire))       
    return  x_train, y_train

x_train, y_train = data_processin(corpus,2)

print(*x_train, sep = '\n')


input_nodes  = len(x_train[0])
hidden_nodes = 15
output_nodes = len(x_train[0])

epochs = 3000
alpha = 0.001


w1 = np.random.random((input_nodes, hidden_nodes))
w2 = np.random.random((hidden_nodes, output_nodes))

for e in range(epochs):
    Loss = 0
    for i in range(len(x_train)):
        #forword propagation
        input_layer  = x_train[i]
        hidden_layer = np.dot(w1.T, input_layer)
        output_layer = softmax(np.dot(w2.T , hidden_layer))
        
        #back propagation
        E = output_layer - y_train[i]
        
        dE_dw2 = np.outer(hidden_layer, E)
        dE_dw1 = np.outer(input_layer, np.dot(w2,E.T)) 
        #updating weight
        w2 -= alpha*dE_dw2
        w1 -= alpha*dE_dw1
        c = 0 
        for j in range(input_nodes):
            if(y_train[i][j]):
                Loss += -1*(np.dot(w2.T , hidden_layer))[j]
                c += 1
                pass
            pass
        Loss += c*np.log(np.sum(np.exp(hidden_layer)))
    if(e%1000 == 0):
        print('epochs ',e,' accuracy','_',' loss',Loss)
        pass
    pass
pass

print("prediction phase")

word = 'first'

one_hot = one_hot_encoding(word, dictionnaire)

print(one_hot)

input_layer  = one_hot
hidden_layer = np.dot(w1.T, input_layer)
output_layer_ = softmax(np.dot(w2.T , hidden_layer))
        
output = dict()

for i in range(input_nodes):
    output[output_layer_[i]] = i
    pass
pass

context = list()

for k in sorted(output, reverse = True):
    context.append(dictionnaire[output[k]])
    if(len(context) >= 3):
        break
    
print(context)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        












