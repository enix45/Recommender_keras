from keras.models import Model
from keras.layers import Input, Concatenate, PReLU, Add
from keras.layers.core import Dense, Dropout, Lambda, Activation

from Custom_Layers import inner_product, outer_product

import numpy as np
import gc

# This model is based on the paper "Product-based Neural Networks for User Response Prediction" in ICDM '16
# https://arxiv.org/abs/1611.00144
# inputs_dim is a list of the dimensions of each input category
# embedding dim is an integer, it is the dimension of the embedding for each category
# prod_dim is an integer, it is the dimension of output of inner-product layer
# hidden_dim is a list of the dimensions of mlp layers
def oPNN(input_dims, embedding_dim, prod_dim, hidden_dims):
	n_cates = len(input_dims)
	x = [Input(shape = (input_dims[i],)) for i in range(n_cates)]
	embs = [Dense(embedding_dim)(x[i]) for i in range(n_cates)]
	lz = Concatenate(axis = -1)(embs)
	lz = Dense(prod_dim)(lz)

	lp = Add()(embs) # f_sigma in the paper
	lp = outer_product(embedding_dim, prod_dim)(lp)

	l = Concatenate(axis = -1)([lp, lz])
	for i in range(len(hidden_dims)):
		l = Dense(hidden_dims[i])(l)
		l = PReLU()(l)
	ans = Dense(1, activation = 'sigmoid')(l)

	return Model(inputs = x, outputs = [ans])
	#return Model(inputs = x, outputs = [lp])

