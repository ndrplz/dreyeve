from keras.optimizers import Adam

# --- GLOBAL --- #
dreyeve_dir = 'Z:/DATA'
dreyeve_train_seq = range(1, 37+1)
dreyeve_test_seq = range(38, 74+1)
n_sequences = 74
total_frames_each_run = 7500

log_dir = 'logs'

# --- TRAIN --- #
batchsize = 3
frames_per_seq = 16
h = 224
w = 224
train_frame_range = range(0, 3500 - frames_per_seq - 1) + range(4000, total_frames_each_run - frames_per_seq - 1)
val_frame_range = range(3500, 4000 - frames_per_seq - 1)
test_frame_range = range(0, total_frames_each_run-frames_per_seq - 1)

frame_size_before_crop = (256, 256)

w_loss_cropped = 0.5
w_loss_fine = 1.0
loss_str = 'mse'
mse_beta = 0.1
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
