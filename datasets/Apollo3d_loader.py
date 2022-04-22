import torch 
import numpy as np 
import cv2 
import scipy 
import os 
import glob
import random
from torch import get_file_path
from torch.utils.data import Dataset, DataLoader
import logging 
logging.basicConfig(level = logging.DEBUG)
import json 

""""
TODO: To decide the shape of the feature vectors as per the pipeline and decisions for the shape of the target tensors as per the loss functions
Dataloader design  
1. Image of the scene 
2. Read the json file for train.json from the data splits and carry out the projections for bev space.
3. Extract camera height putch 


What this dataloader is spitting: 
1. Image of the scene
2. Gt camera_height, camera_pitch 
3. A Tensor with Rho, delta Z, classification score for each tile in the image. 
4. For phis I need to convert the angle values into a probability vector after the softmax. 

Phi: [batch_size, N-tiles, 10]
rho: [batch_size, N-tiles, 1]
delta_z: [batch_size, N-tiles, 1]
classidity_score: [batch_size, N-tiles, 1] or may be [batch_size, N-tiles, 2] as per the loss function used
"""

#create a pytorch dataset class
class Apollo3d_loader(Dataset):
    def __init__(self, camera_intrinsics, data_root, data_splits, phase = "train", transform = False, cfg = None, args = None):
        super(Apollo3d_loader, self,).__init__()
        self.data_root = data_root
        self.camera_intrinsics = camera_intrinsics
        self.data_root = data_root
        self.data_splits = data_splits
        self.phase = phase 
        self.transform = transform
        self.cfg = cfg
        self.args = args 
        
        
        #TODO: separte the initilization of the data_split
        self.gt_filepath = os.path.join(self.data_splits, "test.json")
        self.load_dataset()
        
        """
        TODO: Things I need to add in the args:: for this dataloader
            
            original image height 
            original image width
            ipm_width
            ipm_height
            camera intrinsics
            crop region for the imgae
            if augment the data?? update the projection matrix
            top view region 
        """

    def load_dataset(self):
        
        #NOTE: the splits mentioned on the Geonet may be less than the original dataset
        print("check if gt file path is correct",self.gt_filepath)
        print(os.path.exists(self.gt_filepath))

        lines = []
        # assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
        with open(self.gt_filepath) as f:
            lines_i = f.readlines()
            # print(lines_i)
        f.close()
        lines = lines + lines_i

        print("length of lines",len(lines))


        
        

    #to calculate angle and offsets
    def houghLine(self, image):
        ''' Basic Hough line transform that builds the accumulator array
        Input : image tile (gray scale image)
        Output : accumulator : the accumulator of hough space
                thetas : values of theta (0 : 360)
                rs : values of radius (-max distance : max distance)
        '''

        #Get image dimensions
        # y for rows and x for columns 
        Ny = image.shape[0]
        Nx = image.shape[1]

        #Max diatance is diagonal one 
        Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))

        # Theta in range from -90 to 90 degrees
        thetas = np.deg2rad(np.arange(0, 360))
        
        #Range of radius
        rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)

        #Create accumulator array and initialize to zero
        accumulator = np.zeros((2 * Maxdist, len(thetas)))

        # Loop for each edge pixel
        for y in range(Ny):
            for x in range(Nx):
                # Check if it is an edge pixel
                #  NB: y -> rows , x -> columns
                if image[y,x] > 0:
                    #Loop for each theta
                    # Map edge pixel to hough space
                    for k in range(len(thetas)):
                    #calculate ρ
                        r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    # Increment accumulator at r, theta    
                        # Update the accumulator
                        # N.B: r has value -max to max
                        # map r to its idx 0 : 2*max
                        accumulator[int(r) + Maxdist,k] += 1
        return accumulator, thetas, rs

    def __len__(self):
        pass 


    def __getitem__(self, idx):
        pass


# import re
# import numpy as np 
# import torch

# constant = 2*np.pi/10 ## ie 2*pi/N_angle_bin

# def deg2rad(angle):
#     #convert angle from degree to radian
#     a = (np.pi / 180) 
#     b = angle * a
#     return b 
# def calprob(n_bin, angle):

#     rad_angle = deg2rad(angle)
#     print("rad angle",rad_angle)
#     print("printing abs value",abs((constant * n_bin)- rad_angle))
#     print("again", abs((constant * n_bin)- rad_angle)/constant)
#     # print("checking one more value",1- ((constant * n_bin) - rad_angle)/10)
#     formula = 1 - abs((constant * n_bin) - rad_angle)/constant
    
#     return formula 

# for i in range(10):
    
#     a = calprob(i+1,135) 
#     # print(a)


if __name__ == "__main__":
    #intialize the dataset
    data_root = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/3D_Lane_Synthetic_Dataset/old_data_splits/standard'
    camera_intrinsics = np.array([[1280, 0, 640], [0, 1280, 360], [0, 0, 1]])

    dataset = Apollo3d_loader(camera_intrinsics, data_root, data_splits)

