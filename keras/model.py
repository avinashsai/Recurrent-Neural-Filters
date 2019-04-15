## model.py
import os
import sys
import numpy as np 
import tensorflow as tf
import keras
import keras.backend as K
from keras.engine import Layer 
from keras.layers import Conv1D,GlobalMaxPooling1D
from keras.layers import LSTM,Embedding
from keras.layers import Dropout,TimeDistributed
from keras.layers import Dense,Input
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint

class RnfFilter(Layer):
    def __init__(self,kernelsize,maxlen, **kwargs):
        super(RnfFilter,self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.maxlen = maxlen
        self.cursize = self.maxlen - self.kernelsize + 1
    
    def call(self, x):
        newfilter = []
        for i in range(self.cursize):
            chunk = x[:, i:i+self.kernelsize, :]
            chunk = K.expand_dims(chunk, 1)
            newfilter.append(chunk)
        return K.concatenate(newfilter, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.cursize, self.kernelsize, input_shape[-1])
        

def cnnrnf(embedmatrix,embeddim,kernelsize,hiddendim,embed_drop,dense_drop,recur_drop,pool_drop,maxlen,wordindex,numclasses):

    inp = Input(shape=(maxlen,),dtype='int32')

    embed_layer = Embedding(len(wordindex)+1,embeddim,weights=[embedmatrix],trainable=False)(inp)
    embed_layer = Dropout(embed_drop)(embed_layer)

    rnflayer = RnfFilter(kernelsize,maxlen)(embed_layer)
    time_dis = TimeDistributed(LSTM(hiddendim,return_sequences=False,recurrent_dropout=recur_drop,dropout=dense_drop))(rnflayer)

    pool = GlobalMaxPooling1D()(time_dis)
    pool = Dropout(pool_drop)(pool)

    outp = Dense(numclasses,activation='softmax')(pool)

    model = Model(inputs=inp,outputs=outp)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

def run_model(model,trainind,valind,testind,trainlab,vallab,testlab,batchsize,epochs):

    checkpoint = ModelCheckpoint('rnf.h5', monitor='val_loss', verbose=1, save_best_only=True, 
    save_weights_only=True, mode='min', period=1)

    history = model.fit(trainind,trainlab,validation_data=[valind,vallab],verbose=1,callbacks=[checkpoint],batch_size=batchsize,epochs=epochs)

    test_accuracy = model.evaluate(testind,testlab)[1]

    return test_accuracy