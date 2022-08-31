from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(cfg, model):
    optim_cfg = cfg.optimizer.copy()
    if optim_cfg.type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if optim_cfg.type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
            momentum=optim_cfg.momentum
        )
    elif optim_cfg.type.lower() == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    elif optim_cfg.type.lower() == 'adamw_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.AdamW, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError
    
    return optimizer


def build_scheduler(cfg, optimizer, total_iters_each_epoch, total_epochs, last_epoch):
    optim_cfg = cfg.optimizer.copy()
    

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.type.lower() in ['adam_onecycle', 'adamw_onecycle']:
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.lr, list(optim_cfg.moms), optim_cfg.div_factor, optim_cfg.pct_start
        )
    else:
        decay_steps = [x * total_iters_each_epoch for x in optim_cfg.decay_step_list]
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * optim_cfg.lr_decay
            return max(cur_decay, optim_cfg.lr_clip / optim_cfg.lr)
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.get('lr_warmup', False):
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.warmup_epoch * len(total_iters_each_epoch),
                eta_min=optim_cfg.lr / optim_cfg.div_factor
            )

    return dict(scheduler=lr_scheduler, warmup_scheduler=lr_warmup_scheduler)
