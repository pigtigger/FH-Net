import numpy as np
import torch

from pcflow.utils.data_utils.structures import BaseSceneFlow as BaseSF


class LoadSFFromFile(object):
    def __init__(self,
                 use_dim=[0,1,2],
                 with_fg_mask=False):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.with_fg_mask = with_fg_mask
    
    def __call__(self, data_dict):
        data = np.load(data_dict['path'])
        pc1=data['pc1'][:,self.use_dim].astype(np.float32)
        pc2=data['pc2'][:,self.use_dim].astype(np.float32)
        gt = data['gt'].astype(np.float32)
        if not self.with_fg_mask:
            sf_data = BaseSF(pc1, pc2, gt)
        else:
            fg_mask1 = data['fg_index'].astype(np.bool)
            fg_mask2 = data['fg_index_t'].astype(np.bool)
            sf_data = BaseSF(pc1, pc2, gt, fg_mask1, fg_mask2)
        data_dict['sf_data'] = sf_data
    

        