## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F 

import torch.utils
import torch.utils.data


class temporalvalue(nn.Module):
    def __init__(self,kernelsize,maxlen):
        super(temporalvalue,self).__init__()
        self.kernelsize = kernelsize
        self.maxlen = maxlen
        self.totallen = self.maxlen-self.kernelsize+1

    def forward(self,value):
        semi_filters = []
        for i in range(self.totallen):
            semi_filters.append(torch.unsqueeze(value[:,i:i+self.kernelsize,:],1))
        

        semifilters = torch.cat(semi_filters,1)
        return semifilters
        
class timedistributed(nn.Module):
    def __init__(self,hiddendim,embeddim,drop):
        super(timedistributed,self).__init__()
        self.hiddendim = hiddendim
        self.embeddim = embeddim
        self.drop = drop
        self.drop_inp = nn.Dropout(self.drop)
        self.lstm = nn.LSTM(self.embeddim,self.hiddendim,batch_first=True)
    
    def forward(self,x):
        timesteps = []
        for i in range(x.size(1)):
            inp = self.drop_inp(x[:,i,:,:])
            out,_ = self.lstm(inp,None)
            #print(out.shape)
            timesteps.append(torch.unsqueeze(out[:,-1,:],1))
        finalout = torch.cat(timesteps,1)
        return finalout
        
        
class rnf(nn.Module):
	def __init__(self,embeddim,embedmatrix,kernelsize,maxlen,hiddendim,embeddrop,drop,numclasses):
		super(rnf,self).__init__()

		self.embeddim = embeddim
		self.embedmatrix = embedmatrix
		self.kernelsize = kernelsize
		self.maxlen = maxlen
		self.hiddendim = hiddendim
		self.embeddrop = embeddrop
		self.drop = drop
		self.numclasses = numclasses

		self.embedlayer = nn.Embedding.from_pretrained(self.embedmatrix)
		self.temporal = temporalvalue(self.kernelsize,self.maxlen)
		self.timedist  = timedistributed(self.hiddendim,self.embeddim,self.drop)
		self.dense = nn.Linear(self.maxlen-self.kernelsize+1,self.numclasses)

		self.drop_embed = nn.Dropout(self.embeddrop)
		self.drop_inp = nn.Dropout(self.drop)


	def forward(self,x):
		embedout = self.embedlayer(x)
		embedout = self.drop_embed(embedout)

		cnnrnf = self.temporal(embedout)
		cnnrnf = self.timedist(cnnrnf)
		cnnrnf = torch.max(cnnrnf,2)[0]
		cnnrnf = self.dense(cnnrnf)
	    
		return cnnrnf