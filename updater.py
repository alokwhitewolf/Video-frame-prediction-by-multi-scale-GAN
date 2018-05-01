
from chainer.training import StandardUpdater
import cupy as cp
from loss import l2_loss, gradient_loss, loss_target1, loss_target0
from chainer.functions import resize_images

class Updater(StandardUpdater):
	def __init__(self, *args, **kwargs):
		self.model = kwargs.pop('model')

	def update_core(self):
		data = self.get_iterator('main').next()
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
			loss_gen = loss_target1(dis_output_fake)




