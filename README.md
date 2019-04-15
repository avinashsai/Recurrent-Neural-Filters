# Recurrent-Neural-Filters

# Getting Started

This is the code for the paper Convolutional Neural Networks with Recurrent Neural Filters
 https://www.aclweb.org/anthology/D18-1109. Convolution kernels are simply affine transformations which don't take into account of langauge compositionality. To avoid this problem RNN is used as a kernel instead of linear kernels. In this method, fixed window of words are learned using RNN(LSTM or GRU) taking into account of language long-term dependencies. The words are concatenated in a sequential manner and trained using RNN. Then maxpooling is applied on each timestep followed by a dense layer.
 
 
 # Implemenation
 I have implemented it in PyTorch with exactly same model as in the paper except recurrent dropout on LSTM. I will also add the implementation in Tensorflow.
 
 # How to run (PyTorch)
 
 Download glove 840B https://nlp.stanford.edu/projects/glove/ word vectors and mention corresponding path in main.py
 
 ```
 git clone https://github.com/avinashsai/Recurrent-Neural-Filters.git
 
 cd PyTorch
 
 python main.py <dataset> (sst or sst2)
 
 ```
 
 # How to run (Keras)
 
 Download glove 840B https://nlp.stanford.edu/projects/glove/ word vectors and mention corresponding path in main.py
 
 ```
 git clone https://github.com/avinashsai/Recurrent-Neural-Filters.git
 
 cd keras
 
 python main.py <dataset> (sst or sst2)
 
 ```
 
 The paper is experimented on two datasets SST-1 and SST-2
