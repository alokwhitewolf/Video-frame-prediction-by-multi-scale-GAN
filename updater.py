
from chainer.training import StandardUpdater
import cupy as cp
from loss import l2_loss, gradient_loss, loss_target1, loss_target0
from chainer.functions import resize_images
from chainer import report
from chainer import Variable
from chainer.functions import split_axis
from MultiScaleNetwork import MultiScaleGenerator, MultiScaleDiscriminator


class Updater(StandardUpdater):
	def __init__(self, iterators, optimizers, GeneratorNetwork, DiscriminatorNetwork,
	             g_fmaps, g_k_sizes,
	             d_fmaps, d_k_sizes, d_fc_sizes,
	             device=0,
	             *args, **kwargs):
		"""

		:param iterators:
		:param optimizers:
		:param device:
		:param args:
		:param kwargs:
		"""

		self.GenNetwork = MultiScaleGenerator(g_fmaps, g_k_sizes)
		self.DisNetwork = MultiScaleDiscriminator(d_fmaps, d_k_sizes, d_fc_sizes)

		super(Updater, self).__init__(iterators, optimizers)
		params = kwargs.pop('params')
		self.device = device
		self.LAM_ADV = params['LAM_ADV']
		self.LAM_LP = params['LAM_LP']
		self.LAM_GDL = params['LAM_GDL']

		self._optimizers['GeneratorNetwork'].setup(self.GenNetwork)
		self._optimizers['DiscriminatorNetwork'].setup(self.DisNetwork)


	def update_core(self):
		data = Variable(self.converter(self.get_iterator('main').next(), self.device))
		print(self.device)
		xp = cp.get_array_module(data)
		n, c, h, w = data.shape
		seq, gt = split_axis(data, [c-3], 1)
		del data

		output = None
		total_loss_dis_adv = 0
		total_loss_gen_adv = 0
		for i in range(1, 5):
			if i != 4:
				downscaled_gt = resize_images(gt, (int(h / 2 ** (4 - i)),
				                                   int(w / 2 ** (4 - i))))
				downscaled_seq = resize_images(seq, (int(h / 2 ** (4 - i)),
			                                     int(w / 2 ** (4 - i))))
			else:
				downscaled_gt = gt
				downscaled_seq = seq

			output = self.GenNetwork.singleforward(i, downscaled_seq,
			                                         output)
			dis_output_fake = self.DisNetwork.singleforward(i,output)
			dis_outplut_real = self.DisNetwork.singleforward(i, downscaled_gt)

			loss_dis = (loss_target1(dis_outplut_real) + loss_target0(dis_output_fake)) / 2

			loss_gen = loss_target1(dis_output_fake)

			total_loss_dis_adv += loss_dis
			total_loss_gen_adv += loss_gen

		loss_l2 = l2_loss(output, gt)
		loss_gdl = gradient_loss(output, gt)

		composite_gen_loss = self.LAM_LP*loss_l2 + self.LAM_GDL*loss_gdl + self.LAM_ADV*total_loss_gen_adv
		report({'L2Loss':loss_l2},self.GenNetwork)
		report({'GDL':loss_gdl},self.GenNetwork)
		report({'AdvLoss':total_loss_gen_adv},self.GenNetwork)
		report({'DisLoss':total_loss_dis_adv},self.DisNetwork)
		report({'CompositeGenLoss':composite_gen_loss},self.GenNetwork)

		self.DisNetwork.cleargrads()
		self.GenNetwork.cleargrads()
		composite_gen_loss.backward()
		self._optimizers["GeneratorNetwork"].update()

		self.DisNetwork.cleargrads()
		self.GenNetwork.cleargrads()
		total_loss_dis_adv.backward()
		self._optimizers["DiscriminatorNetwork"].update()






