
import os
from os import path as osp
from pathlib import Path
import glob
import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mmcv import Config, DictAction

import pcflow
from pcflow.utils.init_utils import create_logger, log_config_to_file, set_random_seed
from pcflow.models import build_model
from pcflow.datasets import build_dataloader
from pcflow.utils.train_utils.optimization import build_optimizer, build_scheduler
from pcflow.utils.train_utils import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path.')
    parser.add_argument(
        '--checkpoint_root_dir',
        default='./checkpoints',
        help='the dir to save experiment logs and models, the '
        'final save path is checkpoint-root-dir + exp-group + '
        'exp.')
    parser.add_argument(
        '--exp_group',
        default=None,
        help='experiment group name, the default is config name.')
    parser.add_argument(
        '--exp',
        default='default',
        help='experiment name')
    parser.add_argument(
        '--ckpt', 
        default=None,
        help='the checkpoint file to resume from.')
    parser.add_argument(
        '--pretrained_model', 
        default=None,
        help='the pretrained model weight.')
    parser.add_argument(
        '--use_ddp',
        action='store_true',
        help='whether to use distributed training.')
    parser.add_argument(
        '--epoch',
        default=None,
        help='number of train epochs.')
    parser.add_argument(
        '--workers',
        default=4,
        help='number of workers for dataloader per gpu.')
    parser.add_argument(
        '--batch_size',
        default=None,
        help='If set, it will overwrite `batch_size` or `batch_size_per_gpu` '
        'in config file.')
    parser.add_argument(
        '--test_batch_size',
        default=None,
        help='batch_size for validation. If set, it will overwrite `test_batch_size` '
        'or `test_batch_size_per_gpu` in config file.')
    parser.add_argument(
        '--lr',
        default=None,
        help='learning rate. If set, it will overwrite variable with the same '
        'name in config file.')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.use_ddp = args.use_ddp
    isrank0 = dist.get_rank() == 0 if cfg.use_ddp else True

    # set work_dir 
    checkpoint_root_dir = Path(args.checkpoint_root_dir)
    if args.exp_group is None:
        exp_group = osp.splitext(osp.basename(args.config))[0]
    else:
        exp_group = args.exp_group
    cfg.work_dir = checkpoint_root_dir / exp_group / args.exp
    cfg.ckpt_dir = cfg.work_dir / 'ckpt'
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt is not None:
        cfg.ckpt = args.ckpt
    
    cfg.workers = args.workers
    if args.epoch is not None:
        cfg.epoch = args.epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.test_batch_size is not None:
        cfg.test_batch_size = args.test_batch_size
    if args.lr is not None:
        cfg.lr = args.lr
        cfg.optimizer.lr = args.lr
    
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = cfg.work_dir / f'{timestamp}.log'
    logger = create_logger(log_file, isrank0=isrank0)
    logger.info('**********************Start logging**********************')
    logger.info('OpenPCFlow version: ' + pcflow.__version__)
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    logger.info(f'Distributed training: {cfg.use_ddp}')
    logger.info(f'Batch size per GPU: {cfg.batch_size}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    log_config_to_file(cfg, logger=logger)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    logger.info(f'Set random seed to {cfg.seed}, '
                f'Deterministic: {cfg.deterministic}')
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    # dump config
    cfg.dump(cfg.work_dir / osp.basename(args.config))

    train_set, train_loader = build_dataloader(cfg, mode='train')
    val_set, val_loader = build_dataloader(cfg, mode='val')

    model = build_model(cfg)
    logger.info(f'Model:\n{model}')

    optimizer = build_optimizer(cfg, model)
    
    # load checkpoint if it is possible
    start_epoch = start_iter = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=cfg.use_ddp, logger=logger)

    if args.ckpt is not None:
        start_iter, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=cfg.use_ddp, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(cfg.ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            start_iter, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=cfg.use_ddp, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
    cfg.start_epoch, cfg.start_iter = start_epoch, start_iter

    scheduler = build_scheduler(
        cfg, optimizer, len(train_loader), cfg.epochs, last_epoch=last_epoch)

    # init distributed env first, since logger depends on the dist info.
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if cfg.use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl') 
        device = torch.device("cuda", local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        device = torch.device("cuda")
        model = model.to(device)

    logger.info('**********************Start training **********************')
    train_model(
        cfg=cfg, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        logger=logger, device=device
    )


if __name__ == '__main__':
    main()