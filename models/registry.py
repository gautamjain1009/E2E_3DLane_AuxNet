from utils.registry import Registry, build_from_cfg
import torch.nn as nn

BASELINE = Registry('baseline')
BACKBONES = Registry('backbone')
HEADS =  Registry('heads')
AGGREGATORS = Registry('aggregator')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_baseline(split_cfg, cfg):
    print(BASELINE)
    return build(split_cfg, BASELINE)

def build_backbone(cfg):
    print(BACKBONES)
    split_cfg = cfg.backbone
    return build(split_cfg, BACKBONES)

def build_heads(cfg):
    print(HEADS)
    # split_cfg = cfg.head
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))

def build_aggregator(cfg):
    print(AGGREGATORS)
    split_cfg = cfg.aggregator
    return build(split_cfg, AGGREGATORS, default_args=dict(cfg=cfg))