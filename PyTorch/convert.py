### convert.py
import numpy as np
import collections
from collections import Counter

import torch
from torch.distributions import uniform

def create_vocab(corpus):
	vocabulary = {}

	words = []
	for sentence in corpus:
		words+=sentence.split()

	counts = Counter(words).most_common()

	vocabulary['<PAD>'] = 0
	index = 1
	for word,_ in counts:
		vocabulary[word] = index
		index+=1

	return vocabulary


def load_embed(embed_path):

	embedding_index = {}
	with open(embed_path,'r',encoding='utf-8') as f:
		for line in f.readlines():
			lexicons = line.split()
			word = lexicons[0]
			embedding = torch.from_numpy(np.asarray(lexicons[1:],dtype='float32'))
			embedding_index[word] = embedding
	embed_dim = int(embedding.size()[0])

	return embedding_index,embed_dim


def load_embedmatrix(embed_index,vocab,embeddim):

	embedding_matrix = torch.zeros(len(vocab),embeddim)
	i = 0
	for word in vocab.keys():
		if(word not in embed_index):
			if(word!='<PAD>'):
				embedding_matrix[i,:] = uniform.Uniform(-0.25,0.25).sample(torch.Size([embeddim]))
		else:
			embedding_matrix[i,:] = embed_index[word]
		i+=1

	return embedding_matrix	

def convert_indices(sentence,vocab,maxlen):
	corpusind = [vocab[word] for word in sentence.split() if word in vocab]
	padind = [0]*maxlen
	curlen = len(corpusind)
	if(maxlen-curlen<0):
		padind = corpusind[:maxlen]
	else:
		padind[maxlen-curlen:] = corpusind

	return torch.from_numpy(np.asarray(padind,dtype='int32'))


def get_indices(Xtrain,Xval,Xtest,vocab,maxlen):

	trainind = torch.zeros(len(Xtrain),maxlen)

	for i in range(len(Xtrain)):
		trainind[i] = convert_indices(Xtrain[i],vocab,maxlen)

	valind = torch.zeros(len(Xval),maxlen)

	for i in range(len(Xval)):
		valind[i] = convert_indices(Xval[i],vocab,maxlen)

	testind = torch.zeros(len(Xtest),maxlen)

	for i in range(len(Xtest)):
		testind[i] = convert_indices(Xtest[i],vocab,maxlen)


	return trainind,valind,testind
	
	
def get_loaders(Xtrain,Xval,Xtest,ytrain,yval,ytest,batchsize):
    
	trainarray = torch.utils.data.TensorDataset(Xtrain,ytrain)
	trainloader = torch.utils.data.DataLoader(trainarray,batchsize)
	
	valarray = torch.utils.data.TensorDataset(Xval,yval)
	valloader = torch.utils.data.DataLoader(valarray,batchsize)
	
	testarray = torch.utils.data.TensorDataset(Xtest,ytest)
	testloader = torch.utils.data.DataLoader(testarray,batchsize)
	
	return trainloader,valloader,testloader