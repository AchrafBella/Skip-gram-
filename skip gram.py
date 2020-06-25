# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:53:40 2020

@author: Supernova
"""

import numpy as np


def one_hot_encoding(word, corpus, word_2_id):
    array = np.zeros(len(corpus))
    array[word_2_id[word]] = 1
    return array

def data_processin(corpus, windows_size):
    dictionnaire = corpus.split(' ')
    word_2_id = {w:i for i,w in enumerate(dictionnaire)} 
    id_2_word = {i:w for i,w in enumerate(dictionnaire)}
    
    x_train = list()
    y_train = list()
    
    for i,elm in enumerate(dictionnaire):
        Centre = elm
        x_train.append(one_hot_encoding(Centre, corpus, word_2_id))
        
        for j in range(i-windows_size,i+windows_size + 1):
            y = list()
            if(j != i and j <= len(dictionnaire)-1 and j>= 0):
                y_train.append(one_hot_encoding(dictionnaire[j], corpus, word_2_id))       
    return word_2_id, id_2_word, x_train, y_train




_, _, x_train, y_train = data_processin('hey this was awsome',2)


print(x_train[0])
print('_'*50)
print(*y_train, sep = '\n')