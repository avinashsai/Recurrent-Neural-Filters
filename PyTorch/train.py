## train.py
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 

import torch.utils
import torch.utils.data

import copy

def get_loss(net,loader,device):
	net.eval()
	with torch.no_grad():
		val_loss = 0.0
		for inds,lbs in loader:
			inds = inds.long().to(device)
			lbs = lbs.long().to(device)

			out = net(inds)

			curloss = F.cross_entropy(out,lbs,reduction='sum')
			val_loss+=curloss.item()
		return val_loss/len(loader.dataset)
		
def get_acc(net,loader,device):
	net.eval()
	with torch.no_grad():
		val_acc = 0
		total = 0
		for inds,lbs in loader:
			inds = inds.long().to(device)
			lbs = lbs.long().to(device)

			out = net(inds)

			total+=inds.size(0)
			preds = torch.max(out,1)[1]
			val_acc+=torch.sum(preds==lbs.data).item()

		return (val_acc/total)*100


def trainmodel(model,trainloader,valloader,testloader,numepochs,device):

	optimizer = torch.optim.Adam(model.parameters())
	#criterion = nn.CrossEntropyLoss()

	best_model_wts = copy.deepcopy(model.state_dict())

	for epoch in range(numepochs):
		model.train()
		curloss = 0.0
		val_best_loss = np.Inf
		for indices,labels in trainloader:
			indices = indices.long().to(device)
			labels = labels.long().to(device)

			model.zero_grad()

			output = model(indices)

			loss = F.cross_entropy(output,labels,reduction='sum')
			curloss+=loss.item()

			loss.backward()
			optimizer.step()
		
		valloss = get_loss(model,valloader,device)
		
		if(valloss<val_best_loss):
			val_best_loss = valloss
			best_model_wts = copy.deepcopy(model.state_dict())
		print("Epoch {} Loss {} ".format(epoch+1,valloss))
		
	model.load_state_dict(best_model_wts)

	testacc = get_acc(model,testloader,device)

	return testacc