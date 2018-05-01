
from chainer.training import StandardUpdater
import cupy as cp
from loss import l2_loss, gradient_loss, loss_target1, loss_target0
from chainer.functions import resize_images

# the percentage of the adversarial loss to use in the combined loss
# LAM_ADV = 0.05
# the percentage of the lp loss to use in the combined loss
# LAM_LP = 1
# the percentage of the GDL loss to use in the combined loss
# LAM_GDL = 1


class Updater(StandardUpdater):
	def __init__(self, iterators, optimizers, device=0, *args, **kwargs):
		"""

		:param iterators:
		:param optimizers:
		:param device:
		:param args:
		:param kwargs:
		"""
		params = kwargs.pop('params')
		self.model = kwargs.pop('model')
		self.device = device
		self.LAM_ADV = params['LAM_ADV']
		self.LAM_LP = params['LAM_LP']
		self.LAM_GDL = params['LAM_GDL']
		super(Updater, self).__init__(iterators, optimizers, *args, **kwargs)
		for network_name in self._optimizers:
			self._optimizers[network_name].setup(getattr(self.model, network_name))


	def update_core(self):
		data = self.converter(self.get_iterator('main').next(), self.device)

		xp = cp.get_array_module(data)
		n, c, h, w = data.shape
		seq, gt = xp.split(data, [c-3], 1)
		del data

		output = None
		for i in range(1, 5):
			if i != 4:
				downscaled_gt = resize_images(gt, (int(h / 2 ** (4 - i)),
				                                   int(w / 2 ** (4 - i))))
				downscaled_seq = resize_images(seq, (int(h / 2 ** (4 - i)),
			                                     int(w / 2 ** (4 - i))))
			else:
				downscaled_gt = gt
				downscaled_seq = seq

			output = getattr(self.model, "G"+str(i))(downscaled_seq,
			                                         output)
			loss_l2 = l2_loss(output, downscaled_gt)
			loss_gdl = gradient_loss(output, downscaled_gt)

			dis_output_fake = getattr(self.model, "D"+str(i))(output)
			dis_outplut_real = getattr(self.model, "D"+str(i))(downscaled_gt)

			loss_dis = loss_target1(dis_outplut_real) + loss_target0(dis_output_fake)
			loss_gen = self.LAM_ADV*loss_target1(dis_output_fake) + \
			           self.LAM_GDL*loss_gdl + \
			           self.LAM_LP*loss_l2
			self._optimizers["D" + str(i)].zero_grads()
			loss_dis.backwards()
			self._optimizers["D" + str(i)].update()

			self._optimizers["G" + str(i)].zero_grads()
			loss_gen.backwards()
			self._optimizers["G" + str(i)].update()


