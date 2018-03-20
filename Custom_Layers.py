from keras import backend as K
from keras.engine.topology import Layer

class interaction(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(interaction, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			name = 'kernel',
			shape = (input_shape[1], self.output_dim),
			initializer = 'uniform',
			trainable = True)

	def call(self, x):
		return 0.5 * (K.pow(K.dot(x, self.kernel), 2) - K.dot(K.pow(x, 2), K.pow(self.kernel, 2)))

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

class ans_fm(Layer):
	def __init__(self, sigmoid, **kwargs):
		self.output_dim = 1
		self.sigmoid = sigmoid
		super(ans_fm, self).__init__(**kwargs)

	def call(self, inputs):
		ans = inputs[0] + K.sum(inputs[1], axis = 1)
		if self.sigmoid:
			return K.sigmoid(ans)
		else:
			return ans

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

class dfm(Layer):
	def __init__(self, input_dims, emb_dim, hid_dims, **kwargs):
		self.input_dims = input_dims
		self.emb_dim = emb_dim
		self.hid_dims = hid_dims
		self.n_cates = len(input_dims)
		super(dfm, self).__init__(**kwargs)

	def build(self, input_shape):
		self.emb_kernels = list()
		self.mlp_kernels = list()
		self.mlp_bias = list()
		for i in range(self.n_cates):
			self.emb_kernels.append(
				self.add_weight(
					name = 'emb_' + str(i),
					shape = (self.input_dims[i], self.emb_dim),
					initializer = 'uniform',
					trainable = True))

	def call(self, x):
		embs = [K.dot(x[i], self.emb_kernels[i]) for i in range(self.n_cates)]
		for i in range(self.n_cates):
			if i == 0:
				emb = embs[0]
				q_emb = K.dot(K.pow(x[i], 2), K.pow(self.emb_kernels[i], 2))
			else:
				emb = emb + embs[i]
				q_emb = q_emb + K.dot(K.pow(x[i], 2), K.pow(self.emb_kernels[i], 2))
		fm_ans = 0.5 * (K.pow(emb, 2) - q_emb)
		embs = K.concatenate(embs, axis = -1)
		return K.concatenate([fm_ans, embs], axis = -1)

	def compute_output_shape(self, input_shape):
		#return [(input_shape[0][0], 1), (input_shape[0][0], self.n_cates * self.emb_dim) ]
		return (input_shape[0][0], (self.n_cates + 1) * self.emb_dim)

class ans_dfm(Layer):
	def __init__(self, input_dim, **kwargs):
		self.output_dim = 1
		self.input_dim = input_dim
		super(ans_dfm, self).__init__(**kwargs)
	
	def build(self, input_shape):
		self.kernel = self.add_weight(
			name = 'kernel',
			#shape = (input_shape[1][1], self.output_dim),
			shape = (self.input_dim, self.output_dim),
			initializer = 'uniform',
			trainable = True)
		self.bias = self.add_weight(
			name = 'bias',
			shape = (self.output_dim,),
                        initializer = 'uniform',
			trainable = True)

	def call(self, inputs):
		# Note that the following did not have the linear term of FM component
		ans1 = K.sum(inputs[0], axis = 1, keepdims = True) 
		ans2 = K.bias_add(K.dot(inputs[1], self.kernel), self.bias)
		#return K.sigmoid(ans1 + ans2 + inputs[2])
		return K.sigmoid(ans1 + ans2)

	def compute_output_shape(self, input_shape):
		#return (input_shape[1][0], self.output_dim)
		return (input_shape[0][0], self.output_dim)

class inner_product(Layer):
	def __init__(self, nb_cate, hidden_dim, **kwargs):
		self.nb_cate = nb_cate
		self.hidden_dim = hidden_dim
		super(inner_product, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			name = 'kernel',
			shape = (self.hidden_dim, self.nb_cate),
			initializer = 'uniform',
			trainable = True)

	def call(self, x):
		return K.transpose(K.sum(K.pow(K.dot(self.kernel, K.stack(x, axis = 1)), 2), axis = -1))

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], self.hidden_dim)

class outer_product(Layer):
	def __init__(self, emb_dim, hidden_dim, **kwargs):
		self.emb_dim = emb_dim
		self.hidden_dim = hidden_dim
		super(outer_product, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			name = 'kernel',
			shape = (self.emb_dim ** 2, self.hidden_dim),
			initializer = 'uniform',
			trainable = True)

	def call(self, x):
		# x is f_sigma in the paper
		x = K.expand_dims(x, axis = -1)
		prod = x * K.permute_dimensions(x, (0, 2, 1))
		prod = K.batch_flatten(prod) # prod is now of the shape (batch_size, emb_dim*emb_dim)
		return K.dot(prod, self.kernel)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.hidden_dim)

class cross_layer(Layer):
	def __init__(self, hidden_dim, **kwargs):
		self.hidden_dim = hidden_dim
		super(cross_layer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
                        name = 'kernel',
                        shape = (self.hidden_dim, 1),
                        initializer = 'uniform',
                        trainable = True)
		self.bias = self.add_weight(
                        name = 'bias',
                        shape = (self.hidden_dim, ),
                        initializer = 'uniform',
                        trainable = True)

	def call(self, x):
		return K.bias_add(K.dot(x[1], self.kernel) * x[0] + x[1], self.bias)

	def compute_output_shape(self, input_shape):
		return input_shape

class Gen_prob(Layer):
	def __init__(self, **kwargs):
		#self.nb_item = nb_item
		super(Gen_prob, self).__init__(**kwargs)

	def call(self, x):
		return K.sum(x[0] * x[1], axis = 1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)

class Gen_sim(Layer):
	def __init__(self, **kwargs):
		super(Gen_sim, self).__init__(**kwargs)

	def call(self, x):
		return K.sigmoid(K.sum(x, axis = 1, keepdims = True))

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)




