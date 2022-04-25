import torch 
import numpy as np 
import cv2 
import scipy 
import os 
import glob
import random
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
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

#helper function later to add in the utils.py responsible for the projection 
def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im

def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values

def transform_lane_g2gflat(h_cam, X_g, Y_g, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_gflat = X_g * h_cam / (h_cam - Z_g)
    Y_gflat = Y_g * h_cam / (h_cam - Z_g)

    return X_gflat, Y_gflat

def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals

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
        
        if phase == "train":
            self.data_filepath = os.path.join(self.data_splits, "train.json")
        else :
            self.data_filepath = os.path.join(self.data_splits, "test.json")
        
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

        #TODO: Make it more efficient 
        if not os.path.exists(self.data_filepath):
            raise Exception('Fail to load the train.json file')

        #loading data from the json file
        json_data = [json.loads(line) for line in open(self.data_filepath).readlines()]
        
        #extract image keys from the json file
        self.image_keys = [data['raw_file'] for data in json_data]

        self.data = {l['raw_file']: l for l in json_data}

    #to calculate angle and offsets
    def HoughLine(self, image):
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
        return len(self.image_keys)


    def __getitem__(self, idx):

        batch = {}

        gtdata = self.data[self.image_keys[idx]]
        img_path = os.path.join(self.data_root, gtdata['raw_file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError('cannot find file: {}'.format(img_path))

        img = cv2.imread(img_path)

        gt_camera_height = gtdata['cam_height'] 
        gt_camera_pitch =gtdata['cam_pitch']

        batch.update({"gt_height":gt_camera_height})
        batch.update({"gt_pitch":gt_camera_pitch})
        
        #TODO: correct the data representation of lanelines
        gt_lanelines = gtdata['laneLines']
        batch.update({'gt_lanelines':gt_lanelines})
 
        #TODO: add transforms and update the other data accordingly
        
        #convert the image to tensor
        img = transforms.ToTensor()(img)
        img = img.float()
        batch.update({"image":img})
        
        # return img, gt_camera_height, gt_camera_pitch, gt_lanelines


        """
        TODO: Other Inputs I need to add here except than that,


        """
        return batch
        
def collate_fn(batch):
    """
    This function is used to collate the data for the dataloader
    """
    
    img_data = [item['image'] for item in batch]
    img_data = torch.stack(img_data, dim = 0)
    
    gt_camera_height_data = [item['gt_height'] for item in batch]
    gt_camera_pitch_data = [item['gt_pitch'] for item in batch]
    gt_lanelines_data = [item['gt_lanelines'] for item in batch] #need to check the data representation of lanelines (vis)

    return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data]

# import re
# import numpy as np 
# import torch

# constant = 2*np.pi/10 ## ie 2*pi/N_angle_bin

# def deg2rad(angle):
#     #convert angle from degree to radian
#     a = (np.pi / 180) s
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
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for i, data in enumerate(loader):
        # print(data.keys())
        # print(data['image'].size())
        # print(data['gt_lanelines'].size())
        # print(data['gt_height'])
        # print(data['gt_pitch'])
        print(data[0].shape)
        # print(data[1])
        # print(data[2])
        print(len(data[3][0][1]))## index ,batch_size, i == lane_cnt, data_points
        break
        # print(data)
        # print(i)