## main.py
import os
import sys
import re
import numpy as np

np.random.seed(1332)

from loader import *
from convert import *
from model import *

maxlen = 50
kernelsize = 5
hiddendim=300
embed_drop=0.4
dense_drop=0.4
recur_drop=0.4
pool_drop=0.0
batchsize = 32
epochs = 15

embedpath = '../glove.840B.300d.txt'

if __name__=='__main__':
    dataset = sys.argv[1]

    if(dataset=='sst2'):
        numclasses = 2
    else:
        numclasses = 5

    Xtrain,Xval,Xtest,ytrain,yval,ytest = load_data(dataset)

    embedindex,embeddim = load_embeddings(embedpath)

    tokenizer,wordindex = tokenize(Xtrain)

    embedmatrix = get_embedmatrix(wordindex,embedindex,embeddim)

    trainind,valind,testind = pad(tokenizer,maxlen,Xtrain,Xval,Xtest)

    trainlab,vallab,testlab = get_labels(ytrain,yval,ytest,numclasses)

    rnfmodel = cnnrnf(embedmatrix,embeddim,kernelsize,hiddendim,embed_drop,dense_drop,recur_drop,pool_drop,maxlen,wordindex,numclasses)

    testaccuracy = run_model(rnfmodel,trainind,valind,testind,trainlab,vallab,testlab,batchsize,epochs)

    print("Test Accuracy {} ".format(testaccuracy*100))