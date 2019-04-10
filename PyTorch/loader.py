import numpy as np 
import torch
import string

def process(text):
	text = text.translate(str.maketrans('', '', string.punctuation))
	return text

def load_data(dataset):
	
	Xtrain = []
	trainlab = []
	Xval = []
	vallab = []
	Xtest = []
	testlab = []

	with open('../Datasets/'+dataset+'/train.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			line = process(line[:-1])
			words = line.split()
			label = int(words[0])
			sentence = " ".join(words[1:])

			Xtrain.append(sentence)
			trainlab.append(label)

		ytrain = torch.from_numpy(np.asarray(trainlab,dtype='int32'))


	with open('../Datasets/'+dataset+'/dev.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			line = process(line[:-1])
			words = line.split()
			label = int(words[0])
			sentence = " ".join(words[1:])

			Xval.append(sentence)
			vallab.append(label)

		yval = torch.from_numpy(np.asarray(vallab,dtype='int32'))



	with open('../Datasets/'+dataset+'/test.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			line = process(line[:-1])
			words = line.split()
			label = int(words[0])
			sentence = " ".join(words[1:])

			Xtest.append(sentence)
			testlab.append(label)

		ytest = torch.from_numpy(np.asarray(testlab,dtype='int32'))


	return Xtrain,Xval,Xtest,ytrain,yval,ytest