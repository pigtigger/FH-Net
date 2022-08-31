from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .pipline import __all__ as pip_options
from pcflow.utils.train_utils import scene_flow_EPE


class WaymoSFDataset(Dataset):
    """Waymo scene flow dataset.
    TODO: Add online mode to process data during training to save storage space.
    """
    def __init__(self,
                 cfg,
                 data_path,
                 scene_list=None,
                 pipline=[]):
        super().__init__()
        self.data_path = Path(data_path)
        self.data_infos = []
        self.data_infos = self.get_data_infos(data_path, scene_list)
        self.scene_list = scene_list
        self.frame_map = {}
        self.pipline = []
        for pip_cfg in pipline:
            name = pip_cfg.pop('type')
            self.pipline.append(pip_options[name](**pip_cfg))
    
    def get_data_infos(self, data_path, scene_list):
        data_infos = []
        for s in scene_list:
            path = data_path / '{:03d}'.format(s) / 'sf_data'
            assert path.exists(), 'scene {} is not exist.'.format(s)
            self.frame_map[s] = len(data_infos)
            scene_file_list = sorted(path.glob('*.npz'))
            data_infos += scene_file_list
        return data_infos
    
    def __getitem__(self, index):
        info = self.data_infos[index]
        data_dict = dict(path=info)
        for pip in self.pipline:
            data_dict = pip(data_dict)
        return data_dict
    
    def get_spec_frame(self, scene_id, frame_id):
        return self.__getitem__(self.frame_map[scene_id] + frame_id)
    
    def evaluate_batch(self, data_dict, accumulation):
        flow = data_dict['flow']
        gt = data_dict['gt']
        epe = torch.mean(torch.norm((flow-gt), dim=2))
        epe, acc, acc2, outlier = scene_flow_EPE(flow, gt)
        if accumulation is None or accumulation == {}:
            accumulation = {'EPE_accum':0.0, 'acc_accum':0.0,
                'acc2_accum':0.0, 'outlier':0.0, 'count':0}
        batch_size = gt.shape[0]
        accumulation['EPE_accum'] += epe * batch_size
        accumulation['acc_accum'] += acc * batch_size
        accumulation['acc2_accum'] += acc2 * batch_size
        accumulation['outlier_accum'] += outlier * batch_size
        accumulation['count'] += batch_size
        return accumulation
    
    def get_eval_results(self, accumulation, use_ddp, logger=None):
        epe_accum = accumulation['EPE_accum']
        acc_accum = accumulation['acc_accum']
        acc2_accum = accumulation['acc2_accum']
        outlier_accum = accumulation['outlier_accum']
        count = torch.tensor(accumulation['count']).to(epe_accum.device)
        if use_ddp:
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(epe_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc2_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(outlier_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        epe = epe_accum / count
        acc = acc_accum / count
        acc2 = acc2_accum / count
        outlier = outlier_accum / count
        if logger is not None:
            logger.info('Eval results:' + f'\nEPE: {epe.item()}, Acc: {acc.item()}, ' + \
                f'Acc2: {acc2.item()}, Outlier: {outlier.item()}, total eval frame: {count.item()}')
        return dict(EPE=epe.item(), Acc=acc.item(), Acc2=acc2.item(), Outlier=outlier.item())
    
    @classmethod
    def compare_eval_results(cls, new_res, best_res, key='EPE'):
        is_update = False
        if key in ['EPE', 'Outlier']:
            if best_res is None:
                best_res = np.inf
            if new_res < best_res:
                best_res = new_res
                is_update = True
        elif key in ['Acc', 'Acc2']:
            if best_res is None:
                best_res = 0.0
            if new_res > best_res:
                best_res = new_res
                is_update = True
        else:
            raise NotImplementedError
        return is_update, best_res