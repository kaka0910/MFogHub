method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'van'
hid_S = 64
hid_T = 256
N_T = 8
N_S = 4
# training
epoch = 100
lr = 1e-4
batch_size = 8
val_batch_size = 4
drop_path = 0
sched = 'onecycle'
warmup_epoch = 5