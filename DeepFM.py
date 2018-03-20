from keras.models import Model
from keras.layers import Input, PReLU, Add
from keras.layers.core import Dense, Dropout, Lambda, Activation

from Custom_Layers import ans_dfm, dfm

import numpy as np
import gc

# This model is based on the paper "DeepFM: A Factorization Machine based Neural Network for CTR Prediction" in IJCAI '17
# https://arxiv.org/abs/1703.04247
# inputs_dim is a list of the dimensions of each input category
# embedding dim is an integer, it is the dimension of the embedding for each category
# hidden_dim is a list of the dimensions of mlp layers
def DeepFM(input_dims, embedding_dim, hidden_dims):
	n_cates = len(input_dims)
	#dim = sum(input_dims)
	x = [Input(shape = (input_dims[i],)) for i in range(n_cates)]
	embs = dfm(input_dims = input_dims, emb_dim = embedding_dim, hid_dims = hidden_dims)(x)
	emb = Lambda(lambda x: x[:,embedding_dim:])(embs)
	fm = Lambda(lambda x: x[:,:embedding_dim])(embs)
	
	for i in range(len(hidden_dims)):
		emb = Dense(hidden_dims[i])(emb)
		emb = PReLU()(emb)
	
	# Linear term
	#lin = [Dense(1)(x[i]) for i in range(n_cates)]
	#lin = Add()(lin)

	ans = ans_dfm(hidden_dims[-1])([fm, emb])
	#ans = ans_dfm(hidden_dims[-1])([fm, emb, lin])

	return Model(inputs = x, outputs = ans)
	#return Model(inputs = x, outputs = fm)


