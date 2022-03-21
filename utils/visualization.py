import torch 
import cv2 
import os 
import argparse 
import numpy as np

"""
This class in general will take the model output TODO: further extend it to lane_exist approach
"""
##steps I need to carry out to make the inference or visualization possible 
# 1. Image Preprocessing as per the model trianined on via config 
# 2. Convert the probability maps to lane points in the image space
# 3. Draw those lane points on the image and return the chosen image dont save it. 
# 4. Repeat the above steps for all the images in the folder to be validated on.

# Step:1

class LaneVisualisation(object):

    def __init__(self,cfg):
        self.cfg = cfg

    def preprocess(self,img_path):
        
        r_img = cv2.imread(img_path)
        img = r_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img':img,  'lanes': []}
