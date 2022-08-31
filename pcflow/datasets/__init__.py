import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcflow.utils.data_utils.data_collate import collate
from .kitti_sf_dataset import KittiSFdataset
from .waymo_sf_dataset import WaymoSFDataset


__all__ = {
    'KittiSFdataset': KittiSFdataset,
    'WaymoSFDataset': WaymoSFDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(cfg, mode=True):
    if mode == 'train':
        dataset_cfg = cfg.data.train.dataset.copy()
        dataloader_cfg = cfg.data.train.dataloader.copy()
        if cfg.get('batch_size', False):
            batch_size = cfg.batch_size
        else:
            batch_size = cfg.batch_size_per_gpu
    elif mode == 'val':
        dataset_cfg = cfg.data.val.dataset.copy()
        dataloader_cfg = cfg.data.val.dataloader.copy()
        if cfg.get('test_batch_size', False):
            batch_size = cfg.test_batch_size
        else:
            batch_size = cfg.test_batch_size_per_gpu
    elif mode == 'test':
        dataset_cfg = cfg.data.test.dataset.copy()
        dataloader_cfg = cfg.data.test.dataloader.copy()
        if cfg.get('test_batch_size', False):
            batch_size = cfg.test_batch_size
        else:
            batch_size = cfg.test_batch_size_per_gpu
    else:
        raise NotImplementedError
    dataset_type = dataset_cfg.pop('type')
    dataset = __all__[dataset_type](cfg=cfg, **dataset_cfg)
    
    num_workers = cfg.workers

    if cfg.use_ddp:
        shuffle = dataloader_cfg.pop('shuffle', False)
        if mode == 'train':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            sampler=sampler,
            **dataloader_cfg
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            **dataloader_cfg
        )
    return dataset, dataloader