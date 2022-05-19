#This code is modified from: https://github.com/Turoad/lanedet

import torch 
import cv2 
import os 
import argparse 
import numpy as np
import torch.nn.functional as F

from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

"""
This class in general will take the model output TODO: further extend it to lane_exist approach 
"""
#steps to carry out to make the inference or visualization possible 
# preprocesing and model inference will be done in the train loop
# 1. Image Preprocessing as per the model trianined on via config 
# 2. Convert the probability maps to lane points in the image space
# 3. Draw those lane points on the image and return the chosen image dont save it. 
# 4. Repeat the above steps for all the images in the folder to be validated on.

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self, cfg):
        sample_y = cfg.sample_y
        img_w, img_h = cfg.ori_img_w, cfg.ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration

class LaneVisualisation(object):

    def __init__(self,cfg):
        self.cfg = cfg

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
            for y in self.cfg.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.cfg.thr: 
                    value = np.argmax(line)

                    x = value*self.cfg.ori_img_w/self.cfg.img_width #-1.
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
    
    def to_array(self):
        sample_y = self.cfg.sample_y
        img_w, img_h = self.cfg.ori_img_w, self.cfg.ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
        return lane

    def draw_lines(self, img_path, predictions):

        if not os.path.isfile(img_path):
                raise FileNotFoundError('cannot find image: {}'.format(img_path))
        
        orig_image = cv2.imread(img_path)
        pred_2_lane = self.get_lanes(predictions)[0]
        lanes = [lane.to_array(self.cfg) for lane in pred_2_lane] ## TODO: Verify if working in my case
        
        for lane in lanes:
            for x, y in lane: 
                if x<=0 or y <=0:
                    continue
                x,y = int(x), int(y)
                # print(x,y)
                img = cv2.circle(orig_image, (x, y), 4, (255, 0, 0), 2)
        ## TODO: plot polylines
                
        return img