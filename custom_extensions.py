from chainer import  training
from chainer import serializers

@training.make_extension(trigger=(5000, 'iteration'))
def saveGen(trainer):
	"""

	:param trainer:
	:return:
	"""
	serializers.save_npz('result/GEN.model', trainer.updater.GenNetwork)