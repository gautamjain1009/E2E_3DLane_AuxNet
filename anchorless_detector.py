import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils.helper_functions import *
import torch.nn.functional as F

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

class Anchorless3DLanedetector(nn.Module):
    def __init__(self,cfg, device, input_dim =1):
        super(Anchorless3DLanedetector, self).__init__()

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
        #     
        #NOTE::Will experiment with different modes to verify which will work better.
        #NOTE: In Other modes except non-overlapping it is all (Conv + batch_norm + relu) no MaxPool

        self.bev_encoder = self.make_layers(cfg.encode_config, input_dim, batch_norm=True)
        
        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, self.embedding_dim, 1)
        )

    def make_layers(self, cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], padding=v[2], stride =v[3])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v[0]
        return nn.Sequential(*layers)

    def forward(self,x):
        output = {} 

        cam_height = self.cam_height
        cam_pitch = self.cam_pitch
        
        #spatial transfer features image to ipm
        grid = self.projective_layer(self.M_inv)
        
        # print("checking the shape of grid", grid.shape)
        # print("checking the shape of the input", x.shape)

        x_proj = F.grid_sample(x, grid)
        # print("x_proj.shape: ", x_proj.shape)

        embedding_features = self.embedding(x_proj)
        # print("check the size of the embedding_features: ", embedding_features.shape)

        # Extract the features from the BEV projected grid
        bev_features = self.bev_encoder(x_proj)
        # print("checking the tensor shaoe ater bev_encoder: ", bev_features.shape)
        
        output.update({"embed_out": embedding_features, "bev_out": bev_features})

        return output

    def update_projection(self, cfg, cam_height, cam_pitch):
        print("updating the projection matrix with gt cam_height and cam_pitch")
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        for i in range(cfg.batch_size):
            M, M_inv = homography_im2ipm_norm(cfg.top_view_region, np.array([cfg.org_h, cfg.org_w]),
                                                cfg.crop_y, np.array([cfg.resize_h, cfg.resize_w]),
                                                cam_pitch[i].detach().cpu().numpy(), cam_height[i].detach().cpu().numpy(), cfg.K)
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch

    def update_projection_for_data_aug(self, aug_mats):
        """
            update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
            Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                               2. when cam_height, cam_pitch are online estimated, update H_c for later use
        """
        # if not self.no_cuda:
        #     aug_mats = aug_mats.cuda()

        for i in range(aug_mats.shape[0]):
            # update H_c directly
            self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
            # augmentation need to be applied in unnormalized image coords for M_inv
            aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
            self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])


def load_3d_model(cfg, device, pretrained = False):
    
    if pretrained == True:
        model = Anchorless3DLanedetector(cfg, device)
        model.load_state_dict(torch.load(cfg.MODEL_PATH))
        
    else : 
        model = Anchorless3DLanedetector(cfg, device)
        #reinitialise the weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        print("=> Initialized the anchorless 3d lane detection model weights")
            
    return model 