
from pprint import pprint
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

def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def homography_im2ipm_norm(top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K):
    """
        Compute the normalized transformation such that image region are mapped to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation: 1. invariance in scale change
                                                           2.Torch grid sample is based on normalized grids
    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param org_img_size: the size of original image size: [h, w]
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: [h, w]
    :param cam_pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: H_im2ipm_norm: the normalized transformation from image to IPM image
    """

    # compute homography transformation from ground to image (only this depends on cam_pitch and cam height)
    H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)
    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    # compute top-view corners' coordinates in image
    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    # compute the normalized transformation
    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    # ipm to im
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm

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

def polar_to_catesian(pred_phi, cam_pitch, cam_height, delta_z_pred, rho_pred):
    """
    NOTE: this function is valid only for one tile
    convert the polar coordinates to cartesian coordinates
    :param pred_phi: predicted line angle
    :param cam_pitch: camera pitch
    :param cam_height: camera height
    :param delta_z_pred: predicted delta z
    :param rho_pred: predicted lateral offset
    :return:
    """
    # convert the polar coordinates to cartesian coordinates
    # theta = np.arctan(pred_phi)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(cam_pitch), np.sin(cam_pitch)],
                                [0, -np.sin(cam_pitch), np.cos(cam_pitch)]])
    translation_matrix = np.array([[rho_pred * np.cos(pred_phi), rho_pred * np.sin(pred_phi), delta_z_pred - cam_height]])
    
    cartesian_points = np.dot(rotation_matrix, translation_matrix) # --> (3, 1)

    return cartesian_points
    
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
        self.embedding_dim = cfg.embedding_dim
        self.delta_push = cfg.delta_push
        self.delta_pull = cfg.delta_pull

        org_img_size = np.array([cfg.org_h, cfg.org_w])
        resize_img_size = np.array([cfg.resize_h, cfg.resize_w])
        self.tile_size = cfg.tile_size
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
            nn.Conv2d(8, self.embedding_dim, 1)
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

    def update_projection(self, args, cam_height, cam_pitch):
        print("updating the projection matrix with gt cam_height and cam_pitch")
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        for i in range(self.batch_size):
            M, M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                                args.crop_y, np.array([args.resize_h, args.resize_w]),
                                                cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), args.K)
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch

    def update_projection_for_data_aug(self, aug_mats):
        """
            update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
            Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                               2. when cam_height, cam_pitch are online estimated, update H_c for later use
        """
        if not self.no_cuda:
            aug_mats = aug_mats.cuda()

        for i in range(aug_mats.shape[0]):
            # update H_c directly
            self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
            # augmentation need to be applied in unnormalized image coords for M_inv
            aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
            self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])

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

    def classification_regression_loss(self, rho_pred, rho_gt, delta_z_pred, delta_z_gt, cls_pred, cls_gt, phi_pred, phi_gt ):
        """"
        Params:
            rho_pred: predicted rho [batch_size,13,8]
            rho_gt: ground truth rho [batch_size,13,8]
            delta_z_pred: predicted delta_z [batch_size,13,8]
            delta_z_gt: ground truth delta_z [batch_size,13,8]
            cls_pred: predicted cls [batch_size,13,8]
            cls_gt: ground truth cls [batch_size,13,8]
            phi_pred: predicted phi [batch_size,10,13,8]
            phi_gt: ground truth phi [batch_size,10,13,8]

        return:
            Angle_loss: loss for angle regression (Cross entropy loss for phi vector) + l1 loss for phi vector
            offset_loss: l1 loss for delta_z + l1 loss for rho
            score_loss: BCE loss for cls_score regression

            Overall_loss: score_loss + c_ij * Angle_loss + c_ij * offset_loss
        """

        L1loss= nn.L1Loss()
        BCEloss = nn.BCEWithLogitsLoss()
        CEloss = nn.CrossEntropyLoss()
        
        batch_size = rho_pred.shape[0]

        #VALIDATE & TODO: manage the datatypes later for the loss calculation for training 
        Overall_loss = torch.tensor(0, dtype = cls_pred.dtype, device = cls_pred.device)
        
        #TODO: change the condition for loops(as per the shape of the tile grid)
        for b in range(rho_pred.shape[0]):
            for i in range(rho_pred.shape[1]): # 13 times 
                for j in range(rho_pred.shape[2]): # 8 times
                    #----------------- Offsets loss----------
                    loss_rho_ij = L1loss(rho_pred[b,i,j],rho_gt[b,i,j])
                    loss_delta_z_ij = L1loss(delta_z_pred[b,i,j], delta_z_gt[b,i,j])
                    
                    offsetsLoss_ij = loss_rho_ij + loss_delta_z_ij 
                    
                    #----------------classification score loss---------
                    loss_score_ij = BCEloss(cls_pred[b,i,j], cls_gt[b,i,j])
                    
                    #--------------- Line angle loss ------------------
                    #TODO: add delta phi loss with indicator function
                    loss_phi_ij = CEloss(phi_pred[b,:,i,j].reshape(1,10),phi_gt[b,:,i,j].reshape(1,10))
                    
                    Lineangle_loss = loss_phi_ij
                    
                    #----------------Overall loss -------------------
                    Overall_loss_ij =  loss_score_ij + cls_gt[b,i,j]* Lineangle_loss + cls_gt[b,i,j] * offsetsLoss_ij
                    
                    Overall_loss = Overall_loss + Overall_loss_ij 
            
    #         #TODO: Verify if I need to divid this loss for one batch by grid_w * grid_h
    #         Overall_loss = Overall_loss/ (rho_pred.shape[1]* rho_pred.shape[2])
        Average_OverallLoss = Overall_loss / batch_size
        
        return Average_OverallLoss

    def discriminative_loss(self, embedding, delta_c_gt):  
        
        """
        Embedding == f_ij 
        delta_c = del_i,j for classification if that tile is part of the lane or not
        """
        pull_loss = torch.tensor(0 ,dtype = embedding.dtype, device = embedding.device)
        push_loss = torch.tensor(0, dtype = embedding.dtype, device = embedding.device)
        
        #iterating over batches
        for b in range(embedding.shape[0]):
            
            embedding_b = embedding[b]  #---->(4,H*,W*)
            delta_c_gt_b = delta_c_gt[b] # will be a tensor of size (13,8) or whatever the grid size is consits of lane labels
            
            #delta_c_gt ---> [batch_size, 13, 8] where every element tells you which lane you belong too. 

            ##TODO: Add condition for 0 class
            labels = torch.unique(delta_c_gt_b) #---> array of type of labels
            num_lanes = len(labels)
            
            if num_lanes==0:
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                pull_loss = pull_loss + _nonsense * _zero
                push_loss = push_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_c in labels: # it will run for the number of lanes basically l_c = 1,2,3,4,5 
                
                #1. Obtain one hot tensor for tile class labels
                delta_c = torch.where(delta_c_gt_b==lane_c,1,0) # bool tensor for lane_c ----> size (13,8)
    
                tensor, count = torch.unique(delta_c, return_counts=True)
                N_c = count[1].item() # number of tiles in lane_c
                
                patchwise_mean = []
                
                #extracting tile patches from the embedding tensor
                for r in range(0,embedding_b.shape[1],self.tile_size):
                    for c in range(0,embedding_b.shape[2],self.tile_size):
                        
                        f_ij = embedding_b[:,r:r+self.tile_size,c:c+self.tile_size] #----> (4,32,32) 
                        f_ij = f_ij.reshape(f_ij.shape[0], f_ij.shape[1]*f_ij.shape[2])
                        
                        #2. calculate mean for lane_c (mu_c) patchwise
                        mu_c = torch.sum(f_ij * delta_c[int(r/self.tile_size),int(c/self.tile_size)], dim = 1)/N_c #--> (4) mu for all the four embeddings
                        patchwise_mean.append(mu_c)
                        #3. calculate the pull loss patchwise
                        
                        pull_loss = pull_loss + torch.mean(F.relu( delta_c[int(r/self.tile_size),int(c/self.tile_size)] * torch.norm(f_ij-mu_c.reshape(4,1),dim = 0)- self.delta_pull)**2) / num_lanes
                        
                patchwise_centroid = torch.stack(patchwise_mean) #--> (32*32,4)
                patchwise_centroid = torch.mean(patchwise_centroid, dim =0) #--> (4)
                
                centroid_mean.append(patchwise_centroid)

            centroid_mean = torch.stack(centroid_mean) #--> (num_lanes,4)

            if num_lanes > 1:
                
                #4. calculate the push loss
                centroid_mean_A = centroid_mean.reshape(-1,1, 4)
                centroid_mean_B =centroid_mean.reshape(1,-1, 4)

                dist = torch.norm(centroid_mean_A-centroid_mean_B, dim = 2) #--> (num_lanes,num_lanes)
                dist = dist + torch.eye(num_lanes, dtype = dist.dtype, device = dist.device) * self.delta_push
                
                #divide by 2 to compensate the double loss calculation
                push_loss = push_loss + torch.sum(F.relu(-dist + self.delta_push)**2) / (num_lanes * (num_lanes-1)) / 2
        
        pull_loss= pull_loss / self.batch_size
        push_loss = push_loss / self.batch_size

        return pull_loss, push_loss

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



