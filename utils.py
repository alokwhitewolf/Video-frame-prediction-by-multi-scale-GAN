import numpy as np
from scipy.ndimage import imread
import scipy.misc
from glob import glob
import os
import constants as c
import cupy as cp


def normalize_frames(frames):
	"""
	Convert frames from int8 [0, 255] to float32 [-1, 1].

	param frames: A numpy array. The frames to be converted.
	return: The normalized frames.
	"""
	new_frames = frames.astype(np.float32)
	new_frames /= (255 / 2)
	new_frames -= 1
	return new_frames


def denormalize_frames(frames):
	"""
	Performs the inverse operation of normalize_frames.

	param frames: A numpy array. The frames to be converted.
	return: The denormalized frames.
	"""
	# contingency for gpu calculations
	if not isinstance(frames, (np.ndarray, cp.ndarray)):
		frames = frames.data
	new_frames = frames + 1
	new_frames *= (255 / 2)
	new_frames = new_frames.astype(np.uint8)

	return new_frames


def clip_l2_diff(clip):
	"""
	param clip: A numpy array of shape [1,  3*(c.HIST_LEN + 1), c.TRAIN_HEIGHT, c.TRAIN_WIDTH )].
	return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
	"""
	diff = 0
	# for i in xrange(c.HIST_LEN):
	for i in range(c.HIST_LEN):
		frame = clip[:, 3 * i:3 * (i + 1), : ,: ,]
		next_frame = clip[:, 3 * (i + 1):3 * (i + 2), :, :]
		# noinspection PyTypeChecker
		diff += np.sum(np.square(next_frame - frame))

	return diff


def get_full_clips(data_dir, num_clips, num_rec_out=1):
	"""
	Loads a batch of random clips from the unprocessed test or test data.

	:param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
	:param num_clips: The number of clips to read.
	:param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
						using the previously-generated frames as input. Default = 1.

	:return: An array of shape
			 [num_clips, (3 * (c.HIST_LEN + num_rec_out), c.TRAIN_HEIGHT, c.TRAIN_WIDTH)].
			 A batch of frame sequences with values normalized in range [-1, 1].
	"""

	clips = np.empty([num_clips,
					  (3 * (c.HIST_LEN + num_rec_out)),
					  c.TEST_HEIGHT,
					  c.TEST_WIDTH])
	print("num 0f clips ", num_clips)
	# get random episodes
	ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)
	print('Directory chosen for generating clips: ', ep_dirs)

	# get a random clip of length HIST_LEN + num_rec_out from each episode
	for clip_num, ep_dir in enumerate(ep_dirs):
		ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
		if len(ep_frame_paths) > 4:
			start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
			clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

			# read in frames
			for frame_num, frame_path in enumerate(clip_frame_paths):
				temp_Height, temp_Width = get_train_frame_dims()

				print('Image selected for compression: ', frame_path)
				frame = imread(frame_path, mode='RGB')
				# print('frame: ', frame)
				if temp_Width != c.TEST_WIDTH or temp_Height != c.TEST_HEIGHT:
					# print('in resize')
					frame = scipy.misc.imresize(frame, (c.TEST_HEIGHT, c.TEST_WIDTH))

				# normalize the frames to range the values between -[1,1]
				norm_frame = np.transpose(normalize_frames(frame),(2,0,1))
				print(clips.shape)
				clips[clip_num, frame_num * 3:(frame_num + 1) * 3, :, :] = norm_frame

	return clips


def process_clip():
	"""
	Gets a clip from the test dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

	return: An array of shape [3 * (c.HIST_LEN + 1), c.TRAIN_HEIGHT, c.TRAIN_WIDTH, ].
			 A frame sequence with values normalized in range [-1, 1].
	"""
	# print('process clips for test')
	clip = get_full_clips(c.TRAIN_DIR, 1)
	# print('clip after get_full_clips: ',clip)

	# Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
	# repeat until we have a clip with movement in it.
	take_first = np.random.choice(2, p=[0.95, 0.05])
	cropped_clip = np.empty([1,  3 * (c.HIST_LEN + 1), c.TRAIN_HEIGHT, c.TRAIN_WIDTH])
	# print('processing cropped clips')

	for i in range(100):  # cap at 100 trials in case the clip has no movement anywhere
		crop_x = np.random.choice(c.TEST_WIDTH - c.TRAIN_WIDTH + 1)
		crop_y = np.random.choice(c.TEST_HEIGHT - c.TRAIN_HEIGHT + 1)
		cropped_clip = clip[:, :, crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH]

		if take_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
			break

	# print('cropped_clip: ', cropped_clip)
	return cropped_clip

def get_train_frame_dims():
	# Returns shape of training images
	img_path = glob(os.path.join(c.TRAIN_DIR, '*/*'))[0]
	img = imread(img_path, mode='RGB')
	shape = np.shape(img)

	return shape[0], shape[1]

def get_train_batch():
	"""
	Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

	@return: An array of shape
			[c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
	"""
	clips = np.empty([c.BATCH_SIZE, (3 * (c.HIST_LEN + 1)),c.TRAIN_HEIGHT, c.TRAIN_WIDTH],
					 dtype=np.float32)

	print('batchsize', c.BATCH_SIZE)
	print('test dir clips', c.TRAIN_DIR_CLIPS)
	# for i in xrange(c.BATCH_SIZE):
	for i in range(c.BATCH_SIZE):
		path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS - 1)) + '.npz'
		print('path:', path)
		clip = np.load(path)['arr_0']

		clips[i] = clip

	return clips


def get_test_batch(dir='data/images/test/', seq_len = 4, no_of_preds=3):
	"""

	:param dir: dir of the test batch
	:param seq_len: number of sequences of the input
	:param no_of_preds: number of predictions to be made
	:return:inputframes and the ground truth future frames
	"""
	h, w = c.TEST_HEIGHT , c.TEST_WIDTH
	input_frames = np.empty([1,  3 * (seq_len), h, w])
	future_frames = []
	episode_chosen = np.random.choice(glob('data/images/test/*'))
	print("episode chosen : ",episode_chosen)
	frames = sorted(glob(episode_chosen+"/*"))
	starting_index = np.random.choice(len(frames) - seq_len - no_of_preds)
	for i in range(seq_len):
		print("images in input : ",frames[starting_index+i])
		frame = imread(frames[starting_index+i], mode='RGB')

		if  (frame.shape[0], frame.shape[1]) != (h, w):
			frame = scipy.misc.imresize(frame, (h, w))

		# normalize the frames to range the values between -[1,1]
		norm_frame = np.transpose(normalize_frames(frame),(2,0,1))
		input_frames[0, i* 3:(i + 1) * 3, :, :] = norm_frame

	for j in range(no_of_preds):
		print("images to predict : ",frames[starting_index+seq_len+j])
		frame = imread(frames[starting_index+seq_len+j], mode='RGB')

		if  (frame.shape[0], frame.shape[1]) != (h, w):
			frame = scipy.misc.imresize(frame, (h, w))

		# normalize the frames to range the values between -[1,1]
		norm_frame = np.transpose(normalize_frames(frame),(2,0,1))
		future_frames.append(norm_frame.astype(np.float32))

	return input_frames.astype(np.float32), future_frames
