"""
CU Lane dataloader for binary lane segmentation
"""

import torch 
import cv2 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import random 
import os 
import json

import collections
from .registry import DATASETS, Process

"""
The below code is modified from https://github.com/Turoad/lanedet/blob/main/lanedet/datasets/culane.py
"""

 
LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/test.txt',
    'test': 'list/test.txt',
} 

@DATASETS.register_module
class CULaneLoader(Dataset):

    def __init__(self, data_root, split, transform = False, cfg = None):
        super(CULaneLoader).__init__()        
        self.transform = transform 
        self.data_root = data_root
        self.list_path = os.path.join(data_root, LIST_FILE[split])
        self.split = split
        self.cfg = cfg
        self.process = Process(transform,self.cfg)

        #load annotations for the whole dataset once for iteration for the loader
        self.load_annotations()

    def load_annotations(self):
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos
    
    def binary_segmask(self,mask_i):    
        #assuming there are at max 6 lanes in the dataset

        mask_i[mask_i ==1] = 1
        mask_i[mask_i ==2] = 1
        mask_i[mask_i ==3] = 1
        mask_i[mask_i ==4] = 1
        mask_i[mask_i ==5] = 1
        mask_i[mask_i ==6] = 1
    
        return mask_i 
        
    def __len__(self):
        return len(self.data_infos)

    def __get_item__(self, index):

        batch = {} 

        sample = self.data_infos[index]

        #check if there exists image and mask respectively 
        if not os.path.isfile(sample['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(sample['img_path']))
        

        #load image and mask
        img = cv2.imread(sample['img_path'])
        img = img[self.cfg.cut_height:, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #( BGR -> RGB)
        batch.update({'img':img})

        if self.split == "train" or "val":
            if 'mask_path' in sample and not os.path.isfile(sample['mask_path']):
                raise FileNotFoundError('cannot find file: {}'.format(sample['mask_path']))
        
            label = cv2.imread(sample['mask_path'],cv2.IMREAD_UNCHANGED)

            if len(label.shape) > 2:
                label = label[:, :, 0]
            
            label  = label.squeeze()

            label = label[self.cfg.cut_height:, :]
            batch.update({'mask':label})

        if self.transform:
            batch = self.process(batch)

            binary_mask = self.binary_segmask(batch['mask'])
            batch.update({'binary_mask':binary_mask})

        batch.update({"full_img_path": sample['img_path']})

        return batch 

