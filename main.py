import torch 
import time 
import cv2 

from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 

#build import for different modules
from datasets.registry import build_dataloader


parser = argparse.ArgumentParser(description='2D Lane detection')
parser.add_argument('--config', help = 'path of trian config file')

args = parser.parse_args()

#parasing config file
cfg = Config.fromfile(args.config)

# print(cfg.batch_size)
# print(cfg.img_norm)
# building dataloader
train_loader = build_dataloader(cfg.dataset.train, cfg, is_train = True)

for j in enumerate(train_loader):
    # print(j[1]['mask'].shape)
    # print(j[0])
    print(j[1]['mask'].shape)