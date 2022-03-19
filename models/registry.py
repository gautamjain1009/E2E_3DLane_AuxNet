from utils.registry import Registry, build_from_cfg
import torch.nn as nn

BASELINE = Registry('baseline')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_baseline(split_cfg, cfg):

    return build(split_cfg, BASELINE)
