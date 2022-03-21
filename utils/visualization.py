import torch 
import cv2 
import os 
import argparse 
import numpy as np
import torch.nn.functional as F
from lane import Lane

## test imports
from config import Config

"""
This class in general will take the model output TODO: further extend it to lane_exist approach
"""
#steps I need to carry out to make the inference or visualization possible 
# preprocesing and model inference will be done in the train loop
# 1. Image Preprocessing as per the model trianined on via config 
# 2. Convert the probability maps to lane points in the image space
# 3. Draw those lane points on the image and return the chosen image dont save it. 
# 4. Repeat the above steps for all the images in the folder to be validated on.


class LaneVisualisation(object):

    def __init__(self,cfg, img_path, predictions):
        self.cfg = cfg
        self.img_path = img_path
        self.predictions = predictions

    def get_lanes(self, output):
        segs = output['seg']
        segs = F.softmax(segs, dim=1)
        segs = segs.detach().cpu().numpy()
        
        if 'exist' in output:
            exists = output['exist']
            exists = exists.detach().cpu().numpy()
            exists = exists > 0.5
        else:
            exists = [None for _ in segs]

        ret = []
        for seg, exist in zip(segs, exists):
            lanes = self.probmap2lane(seg, exist)
            ret.append(lanes)
        return ret

    def probmap2lane(self, probmaps, exists=None):
        lanes = []
        probmaps = probmaps[1:, ...]
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.cfg.ori_img_w/self.cfg.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.cfg.ori_img_w
            coord[:, 1] /= self.cfg.ori_img_h
            lanes.append(Lane(coord))
    
        return lanes

    def draw_lines(self, ):
        ## this function will be called in the train script
        ## the output fed to this function 
        ## remember to enable cut height
        image = cv2.imread(self.img_path)
        pred_2_lane = self.get_lanes(self.predictions)
        
        for lane in pred_2_lane:
            for x, y in lane: 
                if x<=0 or y <=0:
                    continue
                x,y = int(x), int(y)
        ## To draw polylines on the image and return the image
                
        pass
        

    