## Main.py
import os
import re
import sys

import time
import numpy as np 
import torch

from loader import *
from convert import *

from model import *
from train import *

embed_path = '../glove.840B.300d.txt'

kernelsize = 5
hiddendim = 300
embeddrop = 0.4
drop = 0.4
maxlen = 50
batchsize = 32
epochs = 20	

if __name__ == '__main__':

	dataset = sys.argv[1]
	
	if(dataset=='sst2'):
	  numclasses = 2
	else:
	  numclasses = 5
	  
	print("Parameters Used : kernelsize {} hiddendim {} Maximum Sentence Length {} NumClasses {} BatchSize {} "
	.format(kernelsize,hiddendim,maxlen,numclasses,batchsize))
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	Xtrain,Xval,Xtest,ytrain,yval,ytest = load_data(dataset)

	vocabulary = create_vocab(Xtrain)

	embeddings,embeddim = load_embed(embed_path)

	embedmatrix = load_embedmatrix(embeddings,vocabulary,embeddim)

	trainind,valind,testind = get_indices(Xtrain,Xval,Xtest,vocabulary,maxlen=50)
	
	model = rnf(embeddim,embedmatrix,kernelsize,maxlen,hiddendim,embeddrop,drop,numclasses).to(device)
	
	#### Testing Model 
	x = torch.ones(4,maxlen).long().to(device)
	out = model(x)
	print("Sample Data Size {} " .format(out.shape))
	
	
	#### Get Loaders
	trainloader,valloader,testloader = get_loaders(trainind,valind,testind,ytrain,yval,ytest,batchsize)
	
	testaccuracy = trainmodel(model,trainloader,valloader,testloader,epochs,device)
	
	print("Test Accuracy {} ".format(testaccuracy))