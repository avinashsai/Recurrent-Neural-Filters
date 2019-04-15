## convert.py
import numpy as np 
import tensorflow as tf 
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_embeddings(embedpath):
    embedding_index = {}
    with open(embedpath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            words = line.split()
            word = words[0]
            vector = np.asarray(words[1:],dtype='float32')

        embed_dim = int(vector.shape[0])

    return embedding_index,embed_dim

def get_embedmatrix(wordindex,embedding_index,embed_dim):
    embedding_matrix = np.zeros((len(wordindex)+1,embed_dim))

    for word,i in wordindex.items():
        embedding_vector = embedding_index.get(word)
        if(embedding_vector is not None):
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.25,0.25,embed_dim)

    return embedding_matrix

def tokenize(Xtrain):
    tok = Tokenizer()
    tok.fit_on_texts(Xtrain)

    return tok,tok.word_index

def pad(tok,maxlen,Xtrain,Xval,Xtest):

    trainX = tok.texts_to_sequences(Xtrain)
    trainind = pad_sequences(trainX,maxlen)

    valX = tok.texts_to_sequences(Xval)
    valind = pad_sequences(valX,maxlen)

    testX = tok.texts_to_sequences(Xtest)
    testind = pad_sequences(testX,maxlen)

    return trainind,valind,testind


def get_labels(ytrain,yval,ytest,numclasses):

    trainlab = keras.utils.to_categorical(ytrain,numclasses)
    vallab = keras.utils.to_categorical(yval,numclasses)
    testlab = keras.utils.to_categorical(ytest,numclasses)

    return trainlab,vallab,testlab