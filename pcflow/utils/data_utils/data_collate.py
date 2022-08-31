
import functools

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs, string_classes

import core.points as custom_data


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)
    return wrapper

class DataContainer(object):
    def __init__(self,
                 data,
                 stack=False,
                 cpu_only=True):
        self._data = data
        self._stack = stack
        self._cpu_only = cpu_only    
        
    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack
    
    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


def collate(batch):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., tensor of constant size
    3. cpu_only = False, stack = False, e.g., tensor of variable size
    """

    if not isinstance(batch, container_abcs.Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        if batch[0].stack and not batch[0].cpu_only:
            stacked = []
            if hasattr(custom_data, batch[0].data.__class__.__name__):
                stacked = type(batch[0].data).stack([sample.data for sample in batch])
            else:
                stacked = default_collate([sample.data for sample in batch])
            return stacked
        else:
            return [sample.data for sample in batch]
    elif isinstance(batch[0], container_abcs.Sequence) and not isinstance(batch[0], string_classes):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(batch[0], container_abcs.Mapping):
        return {
            key: collate([d[key] for d in batch])
            for key in batch[0]
        }
    else:
        return default_collate(batch)