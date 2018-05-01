from chainer.dataset import dataset_mixin
import glob
import numpy as np
from chainer.iterators import SerialIterator
import six


class Dataset(dataset_mixin.DatasetMixin):
	def __init__(self, path):
		self._paths = glob.glob(path+"*.npz")
		self._len = len(self._paths)

	def __len__(self):
		return len(self._paths)

	def __getitem__(self, index):
		if isinstance(index, slice):
			current, stop, step = index.indices(len(self))
			return [self.get_example(i) for i in six.moves.range(current, stop, step)]
		elif isinstance(index, list) or isinstance(index, np.ndarray):
			return [self.get_example(i) for i in index]
		else:
			return self.get_example(index)

	def get_example(self, i):
		"""

		:param i:
		"""
		file_path = self._paths[i % self._len]
		return np.load(file_path)['arr_0'][0]
