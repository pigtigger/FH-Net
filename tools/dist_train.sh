gpuid=0,1,2,3,4,5,6,7

# torch >= 1.9
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpuid} \
torchrun --nproc_per_node 8 --master_port=29500 \
tools/train.py configs/models/flownet3d.py --use_DDP \
--exp first

# torch < 1.9
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpuid} \
# python -m torch.distributed.launch --nproc_per_node 8 --master_port=29500 \
# tools/train.py configs/models/flownet3d.py --use_DDP \
# --exp first