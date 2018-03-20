import numpy as np
import math

from random import shuffle

def split_gen(data, label, s_idx, batch_size):
	nb_data = len(label)
	nb_batch = math.ceil(nb_data / batch_size)
	indices = [i for i in range(nb_data)]
	while True:
		shuffle(indices)
		for i in range(nb_batch):
			idx = indices[i * batch_size:min(nb_data, (i+1) * batch_size)]
			u_feat = data[0][idx].toarray()
			i_feat = data[1][idx].toarray()
			gt = label[idx]
			feat = np.split(u_feat, s_idx[0], axis = 1) 
			feat.extend(np.split(i_feat, s_idx[1], axis = 1))
			yield feat, gt

def multi_gen(data, label, batch_size):
	nb_data = len(label)
	nb_batch = math.ceil(nb_data / batch_size)
	nb_input = len(data)
	#inputs = list()
	while True:
		for i in range(nb_batch):
			for j in range(nb_input):
				if j == 0:
					inputs = [data[j][i * batch_size:min(nb_data, (i+1) * batch_size)].toarray()]
				else:
					inputs.append(data[j][i * batch_size:min(nb_data, (i+1) * batch_size)].toarray())
			gt = label[i * batch_size:min(nb_data, (i+1) * batch_size)]
			yield inputs, gt

def single_gen(data, label, batch_size, shuff = True):
	nb_data = len(label)
	nb_batch = math.ceil(nb_data / batch_size)
	indices = [i for i in range(nb_data)]
	while True:
		if shuff:
			shuffle(indices)
		for i in range(nb_batch):
			idx = indices[i * batch_size:min(nb_data, (i+1) * batch_size)]
			"""
			inputs = data[i * batch_size:min(nb_data, (i+1) * batch_size)].toarray()
			gt = label[i * batch_size:min(nb_data, (i+1) * batch_size)]
			"""
			inputs = data[idx].toarray()
			gt = label[idx]
			yield inputs, gt

def sia_gen(data, label, batch_size, shuff = True):
	nb_data = len(label)
	nb_batch = math.ceil(nb_data / batch_size)
	indices = [i for i in range(nb_data)]
	while True:
		if shuff:
			shuffle(indices)
		for i in range(nb_batch):
			idx = indices[i * batch_size:min(nb_data, (i+1) * batch_size)]
			u_feat = data[0][idx].toarray()
			i_feat = data[1][idx].toarray()
			gt = label[idx]
			yield [u_feat, i_feat], gt

def feat_gen(data, batch_size):
	nb_data = data.shape[0]
	nb_batch = math.ceil(nb_data / batch_size)
	while True:
		for i in range(nb_batch):
			inputs = data[i * batch_size:min(nb_data, (i+1) * batch_size)].toarray()
			yield inputs




