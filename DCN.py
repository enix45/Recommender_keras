from keras.models import Model
from keras.layers import Input, Concatenate, PReLU
from keras.layers.core import Dense, Dropout, Lambda, Activation

from Custom_Layers import cross_layer

import numpy as np
import gc

# This model is based on the paper "Deep & Cross Network for Ad Click Predictions"
# https://arxiv.org/abs/1708.05123
# inputs_dim is a list of the dimensions of each input category
# embedding dim is an integer, it is the dimension of the embedding for each category
# hidden_dim is a list of the dimensions of mlp layers
def DCN(input_dims, embedding_dim, hidden_dims, nb_cross):
	n_cates = len(input_dims)
	dim = n_cates * embedding_dim 
	x = [Input(shape = (input_dims[i],)) for i in range(n_cates)]
	embs = [Dense(embedding_dim)(x[i]) for i in range(n_cates)]
	embs = Concatenate(axis = -1)(embs)
	#l_d = Dense(hidden_dims[0], activation = PReLU())(embs)
	l_d = Dense(hidden_dims[0])(embs)
	l_d = PReLU()(l_d)
	l_c = cross_layer(dim)([embs, embs])
	for i in range(1, len(hidden_dims)):
		#l_d = Dense(hidden_dims[i], activation = PReLU())(l_d)
		l_d = Dense(hidden_dims[i])(l_d)
		l_d = PReLU()(l_d)
	for i in range(1, nb_cross):
		l_c = cross_layer(dim)([embs, l_c])
	ans = Concatenate(axis = -1)([l_d, l_c])
	ans = Dense(1, activation = 'sigmoid')(ans)

	return Model(inputs = x, outputs = [ans])

