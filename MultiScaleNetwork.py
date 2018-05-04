from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer.functions import resize_images
import cupy as cp


def padding_size(k_size):
	"""
	:param k_size: size of the conv kernel
	:return: padding width to have maintain same size of feature
			 maps with kernel size = k_size
	"""
	assert k_size % 2 == 1, "Kernel Size must be odd"
	return int((k_size - 1) / 2)


class SingleScaleGenerator(Chain):
	def __init__(self, fmaps, k_sizes, lowest_scale=False):
		"""
		:param fmaps:
		:param k_sizes:
		:param lowest_scale:
		"""

		self.fmaps = fmaps
		self.k_sizes = k_sizes
		self.lowest_scale = lowest_scale
		super(SingleScaleGenerator, self).__init__()

		net = []
		for i in range(len(self.k_sizes)):
			net += [('conv' + str(i), L.Convolution2D(self.fmaps[i + 1],
													  self.k_sizes[i], stride=1,
													  pad=padding_size(self.k_sizes[i])))]
		with self.init_scope():
			for name, layer in net:
				setattr(self, name, layer)
		self.net = net

	def __call__(self, seq_input, prev_output=None, *args, **kwargs):
		"""

		:param seq_input:
		:param args:
		:param kwargs:
		:return:
		"""
		if not self.lowest_scale:
			# Concatenate scaled image from prev Generator Scale
			scaled_output = resize_images(prev_output, (int(seq_input.shape[2]), int(seq_input.shape[3])))
			seq_input = F.concat((seq_input, scaled_output), 1)

		# Forward Prop
		output = seq_input
		for i in range(len(self.net) - 1):
			output = getattr(self, self.net[i][0])(output)
			output = F.relu(output)

		output = getattr(self, self.net[-1][0])(output)
		output = F.tanh(output)


		return output


class SingleScaleDiscriminator(Chain):
	def __init__(self, fmaps, k_sizes, fc_sizes):
		"""
		:param fmaps:
		:param k_sizes:
		:param fc_sizes:
		"""
		super(SingleScaleDiscriminator, self).__init__()
		self.fmaps = fmaps
		self.k_sizes = k_sizes
		self.fc_sizes = fc_sizes
		net = []

		for i in range(len(k_sizes)):
			net += [('conv' + str(i), L.Convolution2D(self.fmaps[i], self.fmaps[i + 1],
													  self.k_sizes[i], stride=1))]
		for j in range(len(fc_sizes)):
			net += [('fc' + str(j), L.Linear(in_size=None, out_size=fc_sizes[j]))]
		with self.init_scope():
			for name, layer in net:
				setattr(self, name, layer)
		self.net = net

	def __call__(self, x, *args, **kwargs):
		for i in range(len(self.net) - 1):
			x = getattr(self, self.net[i][0])(x)
			x = F.relu(x)
		x = getattr(self, self.net[-1][0])(x)
		x = F.sigmoid(x)
		return x


class MultiScaleGenerator(Chain):
	def __init__(self, g_fmaps, g_k_sizes):
		"""

		:param g_fmaps:
		:param g_k_sizes:
		"""
		super(MultiScaleGenerator, self).__init__()
		self.g_fmaps = g_fmaps
		self.g_k_sizes = g_k_sizes
		assert len(self.g_fmaps) == len(self.g_k_sizes), " No of fmaps and k_sizes must be same"
		self.no_of_scales = len(self.g_fmaps)

		for i in range(self.no_of_scales):
			setattr(self, "G" + str(i + 1),
					SingleScaleGenerator(fmaps=self.g_fmaps[i],
										 k_sizes=self.g_k_sizes[i],
										 lowest_scale=(i == 0)))

	def singleforward(self, scale, seq_input, prev_output=None, *args, **kwargs):
		"""

		:param scale:
		:param seq_input:
		:param prev_output:
		:param args:
		:param kwargs:
		:return:
		"""
		return getattr(self, "G"+str(scale))(seq_input, prev_output=prev_output, *args, **kwargs)

	def predict(self, x, no_of_predictions=1, seq_len=4):
		"""

		:param x:
		:param no_of_predictions:
		:param seq_len:
		:return:
		"""
		# x shape = [n, 12, h, w]
		xp = cp.get_array_module(x)
		n, c, h, w = x.shape
		outputs = []

		for i in range(no_of_predictions):
			seq = resize_images(x, (int(h / 2 ** 3), int(w / 2 ** 3)))
			output = None

			for j in range(1, 5):
				output = self.singleforward(j, seq, output)
				if j != 4:
					seq = resize_images(x, (int(h / 2 ** (3-j)), int(w / 2 ** (3-j))))

			outputs.append(output)
			x = xp.concat([x, output], 1)[:, -seq_len*3:, :, :]
		return outputs

	def __call__(self, *args, **kwargs):
		pass


class MultiScaleDiscriminator(Chain):
	def __init__(self, d_fmaps, d_k_sizes, d_fc_sizes):
		super(MultiScaleDiscriminator, self).__init__()
		self.d_fmaps = d_fmaps
		self.d_k_sizes = d_k_sizes
		self.d_fc_sizes = d_fc_sizes
		assert len(self.d_fmaps) == len(self.d_k_sizes), " No of fmaps and k_sizes must be same"
		self.no_of_scales = len(self.d_fmaps)

		for i in range(self.no_of_scales):
			setattr(self, "D" + str(i + 1), SingleScaleDiscriminator(fmaps=self.d_fmaps[i],k_sizes=self.d_k_sizes[i],fc_sizes=self.d_fc_sizes[i]))

	def singleforward(self, scale, x, *args, **kwargs):
		"""

		:param scale:
		:param self:
		:param x:
		:param args:
		:param kwargs:
		:return:
		"""
		return getattr(self, "D"+str(scale))( x, *args, **kwargs)

	def __call__(self, *args, **kwargs):
		pass
