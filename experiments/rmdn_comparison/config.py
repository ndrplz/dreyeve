DREYEVE_ROOT = 'Z:/DATA'  # local
# DREYEVE_ROOT = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/'  # cineca
frames_per_seq = 16
total_frames_each_run = 7500
encoding_dim = 81920

h, w = 128, 171
T = 50
C = 20
batchsize = 2
lr = 0.0003
hidden_states = 128

nb_epoch = 999
samples_per_epoch = 1024 * batchsize

dreyeve_train_seq = range(1, 37+1)
dreyeve_test_seq = range(38, 74+1)
train_frame_range = range(15, 3500 - T) + range(4000, total_frames_each_run - T)
val_frame_range = range(3500, 4000 - T)
test_frame_range = range(0, total_frames_each_run - T)

