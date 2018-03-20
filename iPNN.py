from keras.models import Model
from keras.layers import Input, Concatenate, PReLU
from keras.layers.core import Dense, Dropout, Lambda, Activation

from Custom_Layers import inner_product

import numpy as np
import gc


# This model is based on the paper "Product-based Neural Networks for User Response Prediction" in ICDM '16
# https://arxiv.org/abs/1611.00144
# inputs_dim is a list of the dimensions of each input category
# embedding dim is an integer, it is the dimension of the embedding for each category
# prod_dim is an integer, it is the dimension of output of inner-product layer
# hidden_dim is a list of the dimensions of mlp layers
def iPNN(input_dims, embedding_dim, prod_dim, hidden_dims):
	n_cates = len(input_dims)
	dim = sum(input_dims)
	x = [Input(shape = (input_dims[i],)) for i in range(n_cates)]
	embs = [Dense(embedding_dim)(x[i]) for i in range(n_cates)]
	lp = inner_product(n_cates, prod_dim)(embs)
	lz = Concatenate(axis = -1)(embs)
	lz = Dense(prod_dim)(lz)
	l = Concatenate(axis = -1)([lp, lz])
	for i in range(len(hidden_dims)):
		#l = Dense(hidden_dims[i], activation = PReLU())(l)
		l = Dense(hidden_dims[i])(l)
		l = PReLU()(l)
	ans = Dense(1, activation = 'sigmoid')(l)
	#ans = Dense(2)(l)

	return Model(inputs = x, outputs = [ans])

if __name__ == "__main__":
	from scipy.sparse import load_npz
	from tools import load_pickle
	from fit_gen import multi_gen, split_gen

	s_idx = [[538, 1067, 1425, 1438, 1445, 1447, 1753, 1760, 2636, 2643, 2818, 2825, 3097],
		[926, 928, 1455, 1811, 1816, 1819, 2372, 2765, 3726, 7086, 7358, 7504, 7516, 8231, 8235, 8381, 8393, 9108]]
	input_dims = [538, 529, 358, 13, 7, 2, 306, 7, 876, 7, 175, 7, 272, 20, 926, 2, 527, 356, 5, 3, 553, 393, 961, 3360, 272, 146, 12, 715, 4, 146, 12, 715, 4]
	
	batch_size = 256
	data_path = 'feature/'
	samp = 'new_hard'
	nb_epoch = 20

	job_train = load_npz(data_path + samp + '_train_job.npz')
	res_train = load_npz(data_path + samp + '_train_res.npz')
	gt_train = np.asarray(load_pickle(data_path + samp + '_train_label.pkl'))
	job_val = load_npz(data_path + samp + '_val_job.npz')
	res_val = load_npz(data_path + samp + '_val_res.npz')
	gt_val = np.asarray(load_pickle(data_path + samp + '_val_label.pkl'))

	print('[INFO] Compiling model...')
	model = iPNN(
		input_dims = input_dims,
		embedding_dim = 128,
		prod_dim = 128,
		hidden_dims = [128, 64, 64, 32])
	model.compile(loss = 'binary_crossentropy',
		optimizer = 'adam',
		metrics = ['acc', 'mae'])
	model.summary()

	print('[INFO] Initiating training')
	model.fit_generator(
		generator = split_gen([job_train, res_train], gt_train, s_idx, batch_size),
		epochs = nb_epoch,
		steps_per_epoch = math.ceil(job_train.shape[0] / batch_size),
		validation_data = split_gen([job_val, res_val], gt_val, s_idx, batch_size),
		validation_steps =math.ceil(job_val.shape[0] / batch_size))

