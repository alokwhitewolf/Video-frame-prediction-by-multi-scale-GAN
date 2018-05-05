import numpy as np
from glob import glob
import constants as c
from utils import process_clip
import argparse as ap



def process_training_data(num_clips):
	"""
	Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
	default.
	:param num_clips: The number of clips to process. Default = 5000000 (set in __main__).
	:warning: This can take a couple of hours to complete with large numbers of clips.
	"""
	num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))

	for clip_num in range(num_prev_clips, num_clips + num_prev_clips):
		clip = process_clip()

		np.savez_compressed(c.TRAIN_DIR_CLIPS + str(clip_num), clip)

		if (clip_num + 1) % 100 == 0: print ('Processed %d clips' % (clip_num + 1))


if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('--clip_num', "-n", default=5, help="Number of clips", type=int)
	args = vars(parser.parse_args())
	process_training_data(args['clip_num'])