from keras.models import Model
from keras.layers import Input, PReLU, Add, Activation
from keras.layers.core import Dense, Dropout

from Custom_Layers import interaction

import numpy as np
import gc

# This model is from "Neural Factorization Machines for Sparse Predictive Analytics"
# https://arxiv.org/abs/1708.05027
# input_dim: dimension of input feature
# embedding_dim: dimension of embeddings
# mlp_dim: a list of dimensions of mlp layers
def NeuFM(input_dim, embedding_dim, mlp_dims):
	x = Input(shape = (input_dim,))
	y1 = Dense(1)(x)
	y2 = interaction(embedding_dim)(x)
	for i in range(len(mlp_dims)):
		#y2 = Dense(mlp_dims[i], activation = PReLU())(y2)
		y2 = Dense(mlp_dims[i])(y2)
		y2 = PReLU()(y2)
	ans = Dense(1)(y2)
	ans = Add()([y1, ans])
	ans = Activation('sigmoid')(ans)
	return Model(inputs = [x], outputs = [ans])

if __name__ == "__main__":
	from fit_gen import single_gen
	from scipy.sparse import load_npz, hstack
	from tools import load_pickle

	s_idx = [[538, 1067, 1425, 1438, 1445, 1447, 1753, 1760, 2636, 2643, 2818, 2825, 3097, 3117], 
		[926, 928, 1455, 1811, 1816, 1819, 2372, 2765, 3726, 7086, 7358, 7504, 7516, 8231, 8235, 8381, 8393, 9108, 9112]]
	
	data_path = 'feature/'
	samp = 'new_hard'
	print('[INFO] Loading data...')
	job_train = load_npz(data_path + samp + '_train_job.npz')
	res_train = load_npz(data_path + samp + '_train_res.npz')
	gt_train = np.asarray(load_pickle(data_path + samp + '_train_label.pkl'))
	job_val = load_npz(data_path + samp + '_val_job.npz')
	res_val = load_npz(data_path + samp + '_val_res.npz')
	gt_val = np.asarray(load_pickle(data_path + samp + '_val_label.pkl'))
	X_train = hstack([job_train, res_train]).tocsr()
	X_val = hstack([job_val, res_val]).tocsr()

	print('[INFO] Compiling model...')
	model = NeuFM(
		input_dim = 12229,
		embedding_dim = 192,
		mlp_dims = [128, 128, 64, 64, 32])
	model.compile(loss = 'binary_crossentropy',
		optimizer = 'adam',
		metrics = ['acc', 'mae'])
	model.summary()

	print('[INFO] Initiating training')
	model.fit_generator(
		generator = single_gen(X_train, gt_train, batch_size),
		epochs = nb_epoch,
		steps_per_epoch = math.ceil(X_train.shape[0] / batch_size),
		validation_data = single_gen(X_val, gt_val, batch_size),
		validation_steps =math.ceil(X_val.shape[0] / batch_size))

