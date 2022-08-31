# include
_base_ = [
    '../datasets/waymo_sf_scene100.py',
]

seed = 1234
deterministic = True

epoch = 150
batch_size_per_gpu = 8
test_batch_size_per_gpu = 8
lr = 0.001

grad_norm_clip=10
ckpt_save_interval=10
max_ckpt_save_num=10
val_interval=1
eval_key_for_save='EPE'

model = dict(
    type='FlowNet3D',
    extra_input_channel=0,
    sa1_npoint=1024,
    sa1_radius=0.5,
    sa1_nsample=16,
    fe_radius=10.0,
    fe_nsample=64,
    su1_nsample=8,
    su1_redius=2.4
)

optimizer = dict(
    type='adamw', # adamw_onecycle if use CycleLR scheduler
    lr=lr,
    weight_decay=0.0001,
    # scheduler settings
    ### StepLR ###
    decay_step_list=list(range(0,epoch,10)),
    lr_decay=0.7,
    lr_clip=1e-5,
    lr_warmup=False,
    warmup_epoch=3,
    div_factor=10,
    ### CycleLR ###
    # moms=[0.95, 0.85],
    # div_factor=10,
    # pct_start=0.4,
)
