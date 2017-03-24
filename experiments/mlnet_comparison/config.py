import math
import uuid

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# number of rows of input images
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))

#########################################################################
# OTHER STUFF                                                           #
#########################################################################
total_frames_each_run = 7500
batchsize = 10
dreyeve_train_seq = range(1, 37+1)
dreyeve_test_seq = range(38, 74+1)
train_frame_range = range(0, 3500) + range(4000, total_frames_each_run)
val_frame_range = range(3500, 4000)
test_frame_range = range(0, total_frames_each_run)
DREYEVE_DIR = 'Z:/DATA'

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# number of validation images
nb_imgs_val = 64 * batchsize
# number of epochs
nb_epoch = 999
# samples per epoch
nb_samples_per_epoch = 256 * batchsize

experiment_id = uuid.uuid4()
