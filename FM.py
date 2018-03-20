from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout

from Custom_Layers import interaction, ans_fm

import numpy as np
import gc

def FM(input_dim, hidden_dim, dropout_rate = 0.2, sigmoid = True):
	x = Input(shape = (input_dim,))
	y1 = Dense(1)(x)
	y2 = interaction(hidden_dim)(x)
	y2 = Dropout(dropout_rate)(y2)
	ans = ans_fm(sigmoid)([y1, y2])
	
	return Model(inputs = [x], outputs = [ans])


