"""
Tu simple dataloader for 2d lane segmentation. Inspired by: https://github.com/Turoad/lanedet/blob/main/lanedet/datasets/tusimple.py
"""

import torch 
import cv2
import logging 
logging.basicConfig(level = logging.DEBUG)
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import random 
import os 
import json

import collections
from .registry import DATASETS
# from transforms import Resize, RandomRotation, RandomHorizontalFlip, Normalize, ToTensor

"""
TODO: Check if dataloading is slow implement a multithreaded loader (Enhancement)
"""
#dataset split for tusimple already defined by the authors

SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

@DATASETS.register_module
class TusimpleLoader(Dataset):
    def __init__(self, data_root, split, transform = False, cfg = None):  
        super(TusimpleLoader).__init__()
        
        self.transform = transform
        self.data_root = data_root 
        self.split = split
        self.cfg = cfg
        self.annotated_files = SPLIT_FILES[self.split]
        self.logger = logging.getLogger('Tusimple_loader')
        self.load_dataset()
        self.logger.info("Tusimple annotations loaded")
        self.size = self.cfg.size
        self.img_norm = self.cfg.img_norm 
        
    def load_dataset(self):

        n_lanes = 0
        self.data = []
        self.logger.info("Loading tu simple dataset")
        
        for file in self.annotated_files:
            annotated_file_path = os.path.join(self.data_root, file)

            with open(annotated_file_path, 'r') as datasource:
                lines = datasource.readlines()
                for line in lines:
                    datasource = json.loads(line)
                    y_samples = datasource['h_samples']
                    lane_gt = datasource['lanes']
                    mask_path = datasource['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                    lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lane_gt]
                    lanes = [lane for lane in lanes if len(lane) > 0]
                    n_lanes = max(n_lanes, len(lanes))
                    self.data.append({
                    'img_path': os.path.join(self.data_root, datasource['raw_file']),
                    'img_name': datasource['raw_file'],
                    'mask_path': os.path.join(self.data_root, mask_path),
                    'lanes': lanes,
                    })
        
        if self.split == 'train' or 'trainval':
            random.shuffle(self.data)
    
    # def compose_transforms(self, batch, size, img_norm):
    
    #     rand_flip = RandomHorizontalFlip()
    #     rand_rotation = RandomRotation()
    #     rsize = Resize(size)    
    #     norm = Normalize(img_norm)
    #     totensor = ToTensor()

    #     batch = rand_flip(batch)
    #     batch = rand_rotation(batch)
    #     batch = rsize(batch)
    #     batch = norm(batch)
    #     batch = totensor(batch)

    #     return batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        batch = {}
        sample = self.data[index]
        
        if not os.path.isfile(sample['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(sample['img_path']))
        
        img = cv2.imread(sample['img_path'])
    
        ## TODO:resizing
        batch.update({'img':img})

        if self.split =="train" or "trainval":
            if not os.path.isfile(sample['mask_path']):
                raise FileNotFoundError('cannot find file: {}'.format(sample['mask_path']))
            
            label = cv2.imread(sample['mask_path'],cv2.IMREAD_UNCHANGED)
            
            if len(label.shape) >2: #reducing the channels in the segmentation mask
                label = label[:,:,0] 
            label = label.squeeze()
            
            ##TODO:resizing
            batch.update({'mask': label})
        
        #TODO: enable lanedata
        # batch.update({'lanes':sample['lanes']})
            
        #augmentation
        if self.transform:
            # print("I was here")
            # batch = self.compose_transforms(batch,self.cfg.size, self.cfg.img_norm)
            batch = batch
        else:
            batch = batch

        return batch ## TODO: Change it if in case the existance is also predicted

# print(DATASETS)
# if __name__ == "__main__":
    #unit test
    # will be added in config
#     root = "/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple" 
#     split = "train"

#     tusimple = TusimpleLoader(root, split, transform= True)
# #     # # print(len(tusimple)) 

# #     # loader = DataLoader(tusimple, batch_size =4, num_workers=1, collate_fn = collate_fn)
#     loader = DataLoader(tusimple, batch_size =4, num_workers=4)
#     print(len(loader))
#     for i,j in enumerate(loader):
#         print(j.keys())
#     #     print(j[1]['mask'].shape)
#     img_norm = dict(
#     mean=[103.939, 116.779, 123.68],
#     std=[1., 1., 1.]
#     )

#     img_height = 368
#     img_width = 640

#     size = (img_height,img_width)
#     rand = np.random.rand(720,1280,3)
#     mask = np.random.rand(720,1280)
#     lanes = [0,203,34,5,6,6,6,6]
    
#     sample = {}
#     sample.update({"img":rand})
#     sample.update({"mask":mask})
#     # sample.update({'lanes':lanes})

#     print(sample.keys())

#     data = compose_transforms(sample,size, img_norm)

#     print(data['img'].shape)
#     print(data['mask'].shape)


