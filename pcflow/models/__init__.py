from .base_model import BaseModel
from .flownet3d import FlowNet3D


__all__ = {
    'BaseModel': BaseModel,
    'FlowNet3D': FlowNet3D
}


def build_model(cfg):
    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')
    model = __all__[model_type](cfg=cfg, **model_cfg)
    return model