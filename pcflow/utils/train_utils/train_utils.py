import glob
import os
from tqdm import tqdm
import time

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_



def train_model(cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                logger,
                device):
    isrank0 = dist.get_rank() == 0 if cfg.use_ddp else True
    start_epoch = cfg.start_epoch
    max_epoch = cfg.epoch
    accumulated_iter = cfg.start_iter
    best_res = None
    
    for epoch in range(start_epoch, max_epoch):
        model.train()
        if cfg.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        loss_record = 0.
        if isrank0:
            pbar = tqdm(total=len(train_loader), desc='train', ncols=100)
        
        for i_iter, batch_data in enumerate(train_loader):
            scheduler.step(accumulated_iter)
            accumulated_iter += 1

            optimizer.zero_grad()
            output = model(batch_data)
            loss_record += output['total_loss'].item()
            output['total_loss'].backward()
            clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            if isrank0:
                pbar.update(1)
        
        if isrank0:
            pbar.close()
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            logger.info(f'epoch {epoch+1}, lr: {cur_lr}, avg_epoch_loss:{loss_record/len(train_loader)}')

            if (epoch + 1) % cfg.ckpt_save_interval == 0:            
                ckpt_list = glob.glob(str(cfg.ckpt_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)
                if len(ckpt_list) >= cfg.max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - cfg.max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])
                ckpt_name = str(cfg.ckpt_dir / ('checkpoint_epoch_%d.pth' % (epoch+1)))
                checkpoint = checkpoint_state(model, optimizer, epoch+1, accumulated_iter)
                torch.save(checkpoint, filename=ckpt_name)
        
        loss_record = 0.

        if cfg.val_interval != -1:
            if (epoch + 1) % cfg.val_interval == 0:
                eval_results = val_model(cfg, model, val_loader, logger, device)
                if isrank0 and cfg.get('eval_key_for_save', None) is not None:
                    save_res = eval_results[cfg.eval_key_for_save]
                    is_update, best_res = val_loader.dataset.compare_eval_results(
                        save_res, best_res, key=cfg.eval_key_for_save)
                    if is_update:
                        ckpt_name = str(cfg.ckpt_dir / ('checkpoint_best_eval.pth'))
                        checkpoint = checkpoint_state(model, optimizer, epoch+1, accumulated_iter)
                        torch.save(checkpoint, filename=ckpt_name) 



@torch.no_grad()
def val_model(cfg,
              model,
              val_loader,
              logger,
              device):
    isrank0 = dist.get_rank() == 0 if cfg.use_ddp else True
    model.eval()
    if isrank0:
        pbar = tqdm(total=len(val_loader), desc='eval', ncols=100)
    loss_record = 0.
    eval_accum = {}
    for i_iter, batch_data in enumerate(val_loader):
        output = model(batch_data)
        loss_record += output['total_loss'].item()
        eval_accum = val_loader.dataset.evaluate_batch(output, eval_accum)
        if isrank0:
            pbar.update(1)
    if isrank0:
        pbar.close()
    eval_results = val_loader.dataset.get_eval_results(eval_accum, cfg.use_ddp, logger)
    return eval_results


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None

    def model_state_to_cpu(model_state):
        model_state_cpu = type(model_state)()  # ordered dict
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    try:
        import pcflow
        version = 'pcdet+' + pcflow.__version__
    except:
        version = 'none'
    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}