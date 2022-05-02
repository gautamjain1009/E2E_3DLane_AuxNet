
from pprint import pprint
from turtle import forward
import torch 
import torch.nn as nn 
import cv2 
from tqdm import tqdm
import os 
# import dotenv
# dotenv.load_dotenv()
import wandb
import time

from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt
from torch.autograd import Variable ##TODO: remove it later only for projective grid

from timing import * 
import logging 
logging.basicConfig(level = logging.DEBUG)

#build import for different modules
from datasets.registry import build_dataloader
from models.build_model import load_model

def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"

def homography_ipmnorm2g(top_view_region):

    """
    homography transformation from IPM normalized to ground corrdinates

    Why these src points are fixed???? what do they mean by that??
    
    """
    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]]) 
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g

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

"""
TODO: Document how this Projective Grid is used (The idea behind it)
"""
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M):
        super().__init__()
        """
        sample grid which will be transformed usim Hipm2img homography matrix.

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        """

        self.N, self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        # use M only to decide the type not value
        self.base_grid = M.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(
                torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        self.base_grid[:, :, :, 1] = torch.ger(
                linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        
        # if device == 'cuda':
        #     self.base_grid = self.base_grid.to(device)

        self.base_grid = self.base_grid.cuda()

    def forward(self, M):

        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa

        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        
        """
        output grid to be used for grid_sample. 
            1. grid specifies the sampling pixel locations normalized by the input spatial dimensions.
            2. pixel locations need to be converted to the range (-1, 1)
        """

        grid = (grid - 0.5) * 2
        return grid

class Anchorless3DLanedetection(nn.Module):
    def __init__(self,cfg, device, input_dim =1):
        super(Anchorless3DLanedetection, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.batch_size =  cfg.batch_size        
        org_img_size = np.array([cfg.org_h, cfg.org_w])
        resize_img_size = np.array([cfg.resize_h, cfg.resize_w])

        cam_pitch = np.pi / 180 * cfg.pitch

        self.cam_height = torch.tensor(cfg.cam_height).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)
        self.cam_pitch = torch.tensor(cam_pitch).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)

        # image scale matrix
        self.S_im = torch.from_numpy(np.array([[cfg.resize_w,              0, 0],
                                               [            0,  cfg.resize_h, 0],
                                               [            0,              0, 1]], dtype=np.float32))
        self.S_im_inv = torch.from_numpy(np.array([[1/np.float(cfg.resize_w),                         0, 0],
                                                   [                        0, 1/np.float(cfg.resize_h), 0],
                                                   [                        0,                         0, 1]], dtype=np.float32))
        self.S_im_inv_batch = self.S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # image transform matrix
        H_c = homography_crop_resize(org_img_size, cfg.crop_y, resize_img_size)
        self.H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # camera intrinsic matrix
        self.K = torch.from_numpy(cfg.K).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # homography ground to camera
        H_g2cam = np.array([[1,                             0,               0],
                            [0, np.sin(-cam_pitch), cfg.cam_height],
                            [0, np.cos(-cam_pitch),               0]])
        self.H_g2cam = torch.from_numpy(H_g2cam).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # transform from ipm normalized coordinates to ground coordinates
        H_ipmnorm2g = homography_ipmnorm2g(cfg.top_view_region)
        self.H_ipmnorm2g = torch.from_numpy(H_ipmnorm2g).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # compute the tranformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im,  M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
        self.M_inv = M_ipm2im

        if self.device == torch.device("cuda"):
            self.M_inv = self.M_inv.cuda()
            self.S_im = self.S_im.cuda()
            self.S_im_inv = self.S_im_inv.cuda()
            self.S_im_inv_batch = self.S_im_inv_batch.cuda()
            self.H_c = self.H_c.cuda()
            self.K = self.K.cuda()
            self.H_g2cam = self.H_g2cam.cuda()
            self.H_ipmnorm2g = self.H_ipmnorm2g.cuda() 

        size_top = torch.Size([self.batch_size, np.int(cfg.ipm_h), np.int(cfg.ipm_w)])

        # ----------------- BEV projective grid -----------------
        self.projective_layer = ProjectiveGridGenerator(size_top, self.M_inv)

        ## ----------------- BEV Encoder -----------------
        self.bev_encoder = self.make_layers([8, 'M', 16, 'M', 32, 'M', 64, "M", 64, "M", 64], input_dim, batch_norm=True)
        
            #matching the bev_encoder features with gt spatial size for regression
        self.layer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(16, 13, kernel_size=1, stride=1, padding=0)
        
        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, cfg.embedding_dim, 1)
        )

    def make_layers(self, cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self,input):

        cam_height = self.cam_height
        cam_pitch = self.cam_pitch
        
        #spatial transfer features image to ipm
        grid = self.projective_layer(self.M_inv)
        x_proj = F.grid_sample(input, grid)
        print("x_proj.shape: ", x_proj.shape)

        embedding_features = self.embedding(x_proj)
        print("check the size of the embedding_features: ", embedding_features.shape)

        # Extract the features from the BEV projected grid
        bev_features = self.bev_encoder(x_proj)
        print("checking the tensor shaoe ater bev_encoder: ", bev_features.shape)
        bev_features = self.layer1(bev_features)
        bev_features = self.layer2(bev_features)
        bev_features = self.layer3(bev_features)

        return bev_features
    """
    Below code needs to activate while training and data augmentation
    """
    # def update_projection(self, args, cam_height, cam_pitch):
    #     print("updating the projection matrix with gt cam_height and cam_pitch")
    #     """
    #         Update transformation matrix based on ground-truth cam_height and cam_pitch
    #         This function is "Mutually Exclusive" to the updates of M_inv from network prediction
    #     :param args:
    #     :param cam_height:
    #     :param cam_pitch:
    #     :return:
    #     """
    #     for i in range(self.batch_size):
    #         M, M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
    #                                             args.crop_y, np.array([args.resize_h, args.resize_w]),
    #                                             cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), args.K)
    #         self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
    #     self.cam_height = cam_height
    #     self.cam_pitch = cam_pitch

    # def update_projection_for_data_aug(self, aug_mats):
    #     """
    #         update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
    #         Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
    #                                            2. when cam_height, cam_pitch are online estimated, update H_c for later use
    #     """
    #     if not self.no_cuda:
    #         aug_mats = aug_mats.cuda()

    #     for i in range(aug_mats.shape[0]):
    #         # update H_c directly
    #         self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
    #         # augmentation need to be applied in unnormalized image coords for M_inv
    #         aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
    #         self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])
    
    """
    DEFINE THE lOSS FUNCTIONS FOR EMBEDDING AND REGRESSION
    """


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("=> Using '{}' for computation ".format(device))

    parser = argparse.ArgumentParser(description="Anchorless 3D Lane Detection Train")
    parser.add_argument("--config", type=str, default="configs/config_anchorless_3dlane.py", help="config file")
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--seed", type=int, default=27, help="random seed")
    parser.add_argument("--baseline", type=bool, default=False, help="enable baseline")

    #parsing args
    args = parser.parse_args()

    #load config file
    cfg = Config.fromfile(args.config)

    # for reproducibility
    torch.manual_seed(args.seed)

    # #trained model paths
    # checkpoints_dir = './nets/3dlane_detection/checkpoints'
    # result_model_dir = './nets/3dlane_detection/model_itr'
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # os.makedirs(result_model_dir, exist_ok=True)


    #dataloader



    #load model
 
    #TODO: Need to match the spatial size of the input image to the 2dmodel and thus the same goes for the 3d model 
    model2d = load_model(cfg, baseline = args.baseline).to(device)
    model3d = Anchorless3DLanedetection(cfg, device).to(device)
    print(model3d)
    #optimizer and scheduler



    #loss functions 


    #training loop
    #unit test
    a = torch.rand(2,3,360,480).to(device)

    o = model2d(a)
    print("checking the shape of the output tensor from 2d lane seg pipeline",o.shape)

    o = o.softmax(dim=1)
    o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
    o = o[:,1:2,:,:]    
    """
    #BIG BIG NOTE: In my case remember I need to obtain the binary seg masks for the other network output 
    currently I have 7 classes Investigate that when the network is finished
    """

    o2 = model3d(o)
    print("checking the shape of the output tensor",o2.shape)





