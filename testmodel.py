from MultiScaleNetwork import MultiScaleGenerator
import constants as c
import chainer
from utils import denormalize_frames, get_test_batch
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

def infer(modelpath='TRAINED_ADVERSARIAL.model', no_of_preds = 7):
	"""
	Visualizes the predictions of the saved model

	:param modelpath: path of the saved Generative model to test on
	:param no_of_preds: No of recursive predcitions to make
	:return:
	"""
	# Initiate the generator model
	model = MultiScaleGenerator(c.SCALE_FMS_G, c.SCALE_KERNEL_SIZES_G)
	chainer.serializers.load_npz(modelpath, model)

	# Get input fot the model as well  as ground truth future frames
	# Here future frames is already a list, NOT an array
	input_frames, ground_truth = get_test_batch(no_of_preds=no_of_preds)

	# Create a split version for visualization
	inputs = np.split(input_frames, [3, 6, 9], 1)

	# Plot the input frames
	fig=plt.figure( figsize=(10, 15))
	for i in range(4):
		f, ax1 = plt.subplots(1, 1)
		# Denormalize
		ax1.imshow(np.transpose(denormalize_frames(inputs[i][0]), (1, 2, 0)))
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.set_xlabel('INPUTS')
		f.savefig("inference/INPUT"+str(i)+'.png', tight=True)
	plt.close()

	# Get prediction
	predictions = model.predict(input_frames, no_of_predictions=no_of_preds)

	# TODO: Visualize the multi scale outputs

	# Create a list of the predicted frames for viz
	for i, frame in enumerate(predictions):
		predictions[i] = np.transpose(denormalize_frames(frame)[0], (1, 2, 0))

	# Create a list of grount truth frames for viz
	for i, frame in enumerate(ground_truth):
		ground_truth[i] = np.transpose(denormalize_frames(frame), (1, 2, 0))

	# Plot Outputs
	fig=plt.figure(figsize=(10, 15))
	for i in range(len(ground_truth)):
		f, ax1 = plt.subplots(1, 1)
		ax1.imshow(predictions[i])
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.set_xlabel('OUTPUTS', color='r')
		f.savefig("inference/OUTPUTS"+str(i)+'.png', tight=True)
	plt.close()

	# Plot comparison
	fig=plt.figure(figsize=(10, 15))
	for i in range(len(predictions)):
		f, (ax1, ax2) = plt.subplots(1, 2)
		ax1.imshow(ground_truth[i])
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.set_xlabel('GROUND TRUTH')
		ax2.imshow(predictions[i])
		ax2.set_xticks([])
		ax2.set_yticks([])
		ax2.set_xlabel('ADVERSARIAL PREDICTIONS', color='r')
		f.savefig('inference/comparison'+str(i)+'.png', tight=True)

if __name__ =="__main__":
	parser = ap.ArgumentParser()
	parser.add_argument('--path', '-p', type=str, default="result/TRAINED_ADVERSARIAL.model")
	parser.add_argument('--no_pred', '-n', type=int, default=7)

	args = parser.parse_args()
	infer(args.path, args.no_pred)