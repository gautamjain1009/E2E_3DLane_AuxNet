import torch 
import cv2
import logging 
logging.basicConfig(level = logging.DEBUG)
import numpy as np 
from torch.utils.data import Dataset
import random 
import os 
import json

#dataset split for tusimple already defined by the authors
SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TusimpleLoader(Dataset):
    def __init__(self, data_root, split, cfg = None):
        super(TusimpleLoader).__init__()
        
        self.data_root = data_root 
        self.split = split
        self.annotated_files = SPLIT_FILES[self.split]
        self.logger = logging.getLogger('Tusimple_loader')
        # self.load_dataset()
        
        # def load_dataset(self):
        self.logger.info("Loading tu simple dataset")
        for file in self.annotated_files:
            annotated_file_path = os.path.join(self.data_root, file)
            print(annotated_file_path)
            with open(annotated_file_path, 'r') as lines:
                print("I am in this loop")
                datasource = lines.readlines()
                for line in lines:
                    datasource = json.loads(line)
                    y_samples = datasource['h_samples']
                    lane_gt = datasource['lanes']
                    mask_path = datasource['raw_file'].replace('clips', 'seg_label')[:-3] + '.png'
                    lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lane_gt]
                    print(lanes)



        if self.split == 'traininig':
            ## shuffle the data for the loader
            pass 

    
    def __len__(self):
        pass 

    
    def __getitem__(self, index):
        pass



if __name__ == "__main__":
    
    root = "/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple" 
    split = "train"
    TusimpleLoader(root, split)