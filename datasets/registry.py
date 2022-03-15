import sys 
sys.path.append('..')
import torch.nn as nn
from utils.registry import Registry, build_from_cfg
from torch.utils.data import DataLoader

DATASETS = Registry('datasets')
AUGMENTATION = Registry('process')

"""
Build all the datasets here and call them in the main function with the config for the approach
"""

"""
adopted from https://github.com/Turoad/lanedet
"""
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

##TODO: Implement it in case I need lane line points for predicting existance
# def collate_fn(instances):
#     batch = []

#     for i in range(len(instances[0])):
#         batch.append([instance[i] for instance in instances])
#     return batch

def build_dataset(split_cfg, cfg):
    print(DATASETS)
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))

def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)
    
    data_loader = DataLoader(dataset, batch_size = cfg.batch_size, shuffle = shuffle,
        num_workers = cfg.workers)

    return data_loader
