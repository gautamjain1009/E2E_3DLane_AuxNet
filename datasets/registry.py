import sys 
sys.path.append('..')
import torch.nn as nn
from utils.registry import Registry, build_from_cfg
from torch.utils.data import DataLoader
import collections

DATASETS = Registry('datasets')
AUGMENTATION = Registry('augmentation')

"""
Build all the datasets here and call them in the main with the config for the approach
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

##TODO: Implement it in case I need lane line points for vis
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


class Process(object):
    """Compose multiple process sequentially.
    Args:
        process (Sequence[dict | callable]): Sequence of process object or
            config dict to be composed.
    """

    def __init__(self, processes, cfg):
        assert isinstance(processes, collections.abc.Sequence)
        self.processes = []
        for process in processes:
            if isinstance(process, dict):
                # print(processes)
                # print(AUGMENTATION)
                process = build_from_cfg(process, AUGMENTATION, default_args=dict(cfg=cfg))
                self.processes.append(process)
            elif callable(process):
                self.processes.append(process)
            else:
                raise TypeError('process must be callable or a dict')

    def __call__(self, data):
        """Call function to apply processes sequentially.
        Args:
            data (dict): A result dict contains the data to process.
        Returns:
           dict: Processed data.
        """
        
        for t in self.processes:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.processes:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
