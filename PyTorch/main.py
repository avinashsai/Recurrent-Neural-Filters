import os
import re
import sys

import time
import numpy as np 
import torch

from loader import *
from convert import *

from model import *

embed_path = '/home/avinashsai/glove.6B/glove.6B.50d.txt'

if __name__ == '__main__':
	dataset = sys.argv[1]

	Xtrain,Xval,Xtest,ytrain,yval,ytest = load_data(dataset)

	vocabulary = create_vocab(Xtrain)

	embeddings,embeddim = load_embed(embed_path)

	embedmatrix = load_embedmatrix(embeddings,vocabulary,embeddim)

	trainind,valind,testind = get_indices(Xtrain,Xval,Xtest,vocabulary,maxlen=50)