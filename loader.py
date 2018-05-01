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
			return np.concatenate([self.get_example(i) for i in six.moves.range(current, stop, step)], 0)
		elif isinstance(index, list) or isinstance(index, np.ndarray):
			return np.concatenate([self.get_example(i) for i in index], 0)
		else:
			return self.get_example(index)

	def get_example(self, i):
		"""

		:param i:
		"""
		file_path = self._paths[i % self._len]
		return np.load(file_path)['arr_0']


class Iterator(SerialIterator):
	def __init__(self, dataset, batch_size, repeat=True, shuffle=True):

		self.dataset = dataset
		self.batch_size = batch_size
		self._repeat = repeat
		self._shuffle = shuffle

		self.reset()

	def __next__(self):
		if not self._repeat and self.epoch > 0:
			raise StopIteration

		self._previous_epoch_detail = self.epoch_detail

		i = self.current_position
		i_end = i + self.batch_size
		N = len(self.dataset)
		batch = self.dataset[i:i_end]
		if i_end >= N:
			if self._repeat:
				rest = i_end - N
				if self._order is not None:
					np.random.shuffle(self._order)
				if rest > 0:
					if self._order is None:
						batch.extend(self.dataset[:rest])
					else:
						batch.extend([self.dataset[index] for index in self._order[:rest]])
				self.current_position = rest
			else:
				self.current_position = 0

			self.epoch += 1
			self.is_new_epoch = True
		else:
			self.is_new_epoch = False
			self.current_position = i_end

		return batch

	next = __next__



