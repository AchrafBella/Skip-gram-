# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:22:43 2020

@author: Supernova
"""
from scipy.special import softmax
import numpy as np

class CBOW():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, alpha):
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
                
        self.alpha = alpha
        
        self.w1 = np.random.random((input_nodes, hidden_nodes))
        self.w2 = np.random.random((hidden_nodes, output_nodes))
        pass
    
    def forword(self, x):
        input_layer  = x
        hidden_layer = np.dot(self.w1.T, input_layer)
        output_layer = softmax(np.dot(self.w2.T , hidden_layer))
        return output_layer, hidden_layer
    
    def backpropagation(self,input_layer, hidden_layer, output_layer, target):
        #back propagation
        E = output_layer - target
        
        dE_dw2 = np.outer(hidden_layer, E)
        dE_dw1 = np.outer(input_layer, np.dot(self.w2,E.T)) 
        #updating weight
        self.w2 -= self.alpha*dE_dw2
        self.w1 -= self.alpha*dE_dw1
        pass
    
    def fit(self, x_train, y_train, epochs):
        for e in range(epochs):
            Loss = 0
            for i in range(len(x_train)):
                output_layer, hidden_layer = self.forword(x_train[i])
                self.backpropagation(x_train[i], hidden_layer, output_layer, y_train[i])
                c = 0
                for j in range(self.input_nodes):
                     if(y_train[i][j]):
                         Loss += -1*(np.dot(self.w2.T , hidden_layer))[j]
                         c += 1
            Loss += c*np.log(np.sum(np.exp(hidden_layer)))
            if(e % 1000):
                print('epochs ',e,' accuracy','_',' loss',Loss)
            pass
        pass
    
    def predict(self, word_coded, dictionnaire, contexte_word_num):
        output_layer, _ = self.forword(word_coded)
        output = {}
        for i in range(self.input_nodes):
            output[output_layer[i]] = i
            pass
        context = list()
        for k in sorted(output, reverse = True):
            context.append(dictionnaire[output[k]])
            if(len(context) >= contexte_word_num):
                break
        return context
    
    

corpus = "Well Tbh he wasn't open minded bzzf And he cared about what ppl say....so i felt like i can't do whatever i want mli kankun m3ah Which is not in my nature Anim a person who doesn't care about other ppl opinions about me... Tani 7aja he didn't support my dreams and said some hurtful things and made me feel bad....O kan dima kaykhalini n7ass brassi ana li ghalta Saraha he made me feel bad to myself and i hated that.... He was very toxic O kayt7assab bzzf"
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
unique_words = len(dictionnaire)
cbow = CBOW(unique_words, 50, unique_words, 0.001)
cbow.fit(x_train, y_train, 2000)
word_coded = one_hot_encoding('open', dictionnaire)
pre = cbow.predict(word_coded, dictionnaire, 4)
print(pre)








