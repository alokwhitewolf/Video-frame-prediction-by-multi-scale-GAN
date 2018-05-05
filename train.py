import constants as c
from MultiScaleNetwork import MultiScaleDiscriminator, MultiScaleGenerator
import chainer
from updater import Updater
from loader import Dataset
from chainer.training import extensions
from custom_extensions import saveGen
import argparse

def main(resume, gpu, load_path, data_path):
	dataset = Dataset(data_path)


	GenNetwork = MultiScaleGenerator(c.SCALE_FMS_G, c.SCALE_KERNEL_SIZES_G)
	DisNetwork = MultiScaleDiscriminator(c.SCALE_CONV_FMS_D, c.SCALE_KERNEL_SIZES_D, c.SCALE_FC_LAYER_SIZES_D)

	optimizers = {}
	optimizers["GeneratorNetwork"] = chainer.optimizers.SGD(c.LRATE_G)
	optimizers["DiscriminatorNetwork"] = chainer.optimizers.SGD(c.LRATE_D)

	iterator = chainer.iterators.SerialIterator(dataset, 1)
	params = {'LAM_ADV': 0.05, 'LAM_LP': 1, 'LAM_GDL': .1}
	updater = Updater(iterators=iterator, optimizers=optimizers,
	                  GeneratorNetwork=GenNetwork,
	                  DiscriminatorNetwork=DisNetwork,
	                  params=params,
	                  device=gpu
	                  )
	if gpu>=0:
		updater.GenNetwork.to_gpu()
		updater.DisNetwork.to_gpu()

	trainer = chainer.training.Trainer(updater, (500000, 'iteration'), out='result')
	trainer.extend(extensions.snapshot(filename='snapshot'), trigger=(1, 'iteration'))
	trainer.extend(extensions.snapshot_object(trainer.updater.GenNetwork, "GEN"))
	trainer.extend(saveGen)

	log_keys = ['epoch', 'iteration', 'GeneratorNetwork/L2Loss', 'GeneratorNetwork/GDL',
	            'DiscriminatorNetwork/DisLoss', 'GeneratorNetwork/CompositeGenLoss']
	print_keys = ['GeneratorNetwork/CompositeGenLoss','DiscriminatorNetwork/DisLoss']
	trainer.extend(extensions.LogReport(keys=log_keys, trigger=(10, 'iteration')))
	trainer.extend(extensions.PrintReport(print_keys), trigger=(10, 'iteration'))
	trainer.extend(extensions.PlotReport(['DiscriminatorNetwork/DisLoss'], 'iteration', (10, 'iteration'), file_name="DisLoss.png"))
	trainer.extend(extensions.PlotReport(['GeneratorNetwork/CompositeGenLoss'], 'iteration', (10, 'iteration'), file_name="GenLoss.png"))
	trainer.extend(extensions.PlotReport(['GeneratorNetwork/AdvLoss'], 'iteration', (10, 'iteration'), file_name="AdvGenLoss.png"))
	trainer.extend(extensions.PlotReport(['GeneratorNetwork/AdvLoss','DiscriminatorNetwork/DisLoss'], 'iteration', (10, 'iteration'), file_name="AdversarialLosses.png"))
	trainer.extend(extensions.PlotReport(['GeneratorNetwork/L2Loss'], 'iteration', (10, 'iteration'),file_name="L2Loss.png"))
	trainer.extend(extensions.PlotReport(['GeneratorNetwork/GDL'], 'iteration', (10, 'iteration'),file_name="GDL.png"))

	trainer.extend(extensions.ProgressBar(update_interval=10))
	if resume:
		# Resume from a snapshot
		chainer.serializers.load_npz(load_path, trainer)
	print(trainer.updater.__dict__)
	trainer.run()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume', '-r', type=int, default=0)
	parser.add_argument('--load', '-l', type=str, default="result/snapshot")
	parser.add_argument('--gpu', '-g', type=int, default=-1)
	parser.add_argument('--data', '-d', type=str, default="data/trainclips")

	args = parser.parse_args()
	print(args)
	main(resume = args.resume, gpu=args.gpu,
	    load_path=args.load, data_path=args.data)



