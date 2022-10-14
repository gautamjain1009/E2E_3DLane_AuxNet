import sys
import time
sys.path.append("../")
import torch 
import numpy as np 
import cv2 
import os 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json 
from utils.helper_functions import *
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image
import torchvision.transforms.functional as F

"""
Import just for unit test
"""
from utils.config import Config

#TODO: move this class to utils during code cleanup later
class Visualization(object):
    def __init__(self, org_h, org_w, resize_h, resize_w, K, ipm_w, ipm_h, crop_y, top_view_region):
        
        self.min_y = 0
        self.max_y = 80
        
        self.K = K
        
        self.crop_y = crop_y
        self.top_view_region = top_view_region

        """
            TODO: change the org_h and org_w according to feature map size in the network may be top view region too
        """
        self.org_h = org_h
        self.org_w = org_w
        self.resize_h = resize_h
        self.resize_w = resize_w
        
        self.ipm_w = ipm_w
        self.ipm_h = ipm_h

        self.H_crop = homography_crop_resize([self.org_h, self.org_w], self.crop_y, [self.resize_h, self.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(self.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)

        self.x_min = top_view_region[0, 0]
        self.x_max = top_view_region[1, 0]

        self.y_samples = np.linspace(self.min_y, self.max_y, num=100, endpoint=False)
        
        self.colors = [[1, 0, 0],  # red
          [0, 1, 0],  # green
          [0, 0, 1],  # blue
          [1, 0, 1],  # purple
          [0, 1, 1],  # cyan
          [1, 0.7, 0]]  # orange

    def draw_lanes(self, gt_lanes, img, gt_cam_height, gt_cam_pitch):
        
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)

        H_g2im = homography_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        # print("Checking the shape of original image: ", img.shape)

        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        # print("Checking the shape of cropped image: ", img.shape)     

        # img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h)) 
        # im_ipm = np.clip(im_ipm, 0, 1)
        cnt_gt = len(gt_lanes)
        gt_visibility_mat = np.zeros((cnt_gt, 100))
                
        # resample gt and pred at y_samples
        # for i in range(cnt_gt):
        #     min_y = np.min(np.array(gt_lanes[i])[:, 1])
        #     max_y = np.max(np.array(gt_lanes[i])[:, 1])
        #     x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            
        #     gt_lanes[i] = np.vstack([x_values, z_values]).T
        #     gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
        #                                                np.logical_and(x_values <= self.x_max,
        #                                                               np.logical_and(self.y_samples >= min_y,
        #                                                                              self.y_samples <= max_y)))
        #     gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        flag = False     
        dummy_image = im_ipm.copy()
        
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        #removing axis lables
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])

        for i in range(cnt_gt):
            x_values = np.array(gt_lanes[i])[:, 0]
            y_values = np.array(gt_lanes[i])[:, 1]
            z_values = np.array(gt_lanes[i])[:, 2]
            
            # TODO: remove this condition later
            if flag == True:
                x_ipm_values, y_ipm_values = transform_lane_g2gflat(gt_cam_height, x_values[:100], self.y_samples, z_values[:100])
                # remove those points with z_values > gt_cam_height, this is only for visualization on top-view
                x_ipm_values = x_ipm_values[np.where(z_values[:100] < gt_cam_height)]
                y_ipm_values = y_ipm_values[np.where(z_values[:100] < gt_cam_height)]

            else:
                x_ipm_values = x_values
                y_ipm_values = y_values

            #for vis on IPM image
            x_ipm_values, y_ipm_values = homographic_transformation(self.H_g2ipm, x_ipm_values, y_ipm_values)
            x_ipm_values = x_ipm_values.astype(np.int)
            y_ipm_values = y_ipm_values.astype(np.int)

            #for vis on original image
            x_2d, y_2d = projective_transformation(P_gt, x_values, y_values, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)

            #draw on 2d image
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                # if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k]:
                    #check if the point is in the image
                if 0 <= x_2d[k] <= img.shape[1] and 0<= y_2d[k] <= img.shape[0] and  0 <= x_2d[k-1] <= img.shape[1] and 0 <= y_2d[k-1] <= img.shape[0]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [255,0,0], 3)

            # draw on ipm
            for k in range(1, x_ipm_values.shape[0]):
                # print("first loop", vis)
                # only draw the visible portion
                # if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k] and z_values[k] < gt_cam_height:
                    # print("print here", vis)
                    #TODO: Verify - or + for the value for delta_z
                dummy_image = cv2.line(dummy_image, (x_ipm_values[k - 1], y_ipm_values[k - 1]),
                            (x_ipm_values[k], y_ipm_values[k]), (255,0,0), 1)

            # # # draw in 3d
            # ax1.plot(x_values[np.where(gt_visibility_mat[i, :])],
            #         self.y_samples[np.where(gt_visibility_mat[i, :])],
            #         z_values[np.where(gt_visibility_mat[i, :])], color= 'green', linewidth=1)
            ax1.plot(x_values,
                    y_values,
                    z_values, color= 'green', linewidth=1)


        ax2.imshow(img[:,:,[2,1,0]])
        ax3.imshow(dummy_image[:,:,[2,1,0]])
        #save the plot as image
        # plt.savefig('test.png')

        return fig

class CalculateDistanceAngleOffests(object):
    def __init__(self, org_h, org_w, resize_h, resize_w, K, ipm_w, ipm_h, crop_y, top_view_region):
        
        self.min_y = 0
        self.max_y = 80
        
        self.K = K
        
        self.crop_y = crop_y
        self.top_view_region = top_view_region

        """
            TODO: change the org_h and org_w according to feature map size in the network may be top view region too
        """
        self.org_h = org_h
        self.org_w = org_w
        self.resize_h = resize_h
        self.resize_w = resize_w
        
        self.ipm_w = ipm_w
        self.ipm_h = ipm_h

        self.H_crop = homography_crop_resize([self.org_h, self.org_w], self.crop_y, [self.resize_h, self.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(self.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)

        self.x_min = top_view_region[0, 0]
        self.x_max = top_view_region[1, 0]

        self.y_samples = np.linspace(self.min_y, self.max_y, num=100, endpoint=False)

    def draw_bevlanes(self, gt_lanes, img, gt_cam_height, gt_cam_pitch, color_list, vis = False):
        
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)

        H_g2im = homography_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        # print("Checking the shape of original image: ", img.shape)

        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        # cv2.imwrite("report_image.jpg", img)
        # print("Checking the shape of cropped image: ", img.shape)     

        # img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
        # cv2.imwrite("report_image_ipm.jpg", im_ipm)
        # im_ipm = np.clip(im_ipm, 0, 1)
        cnt_gt = len(gt_lanes)
        gt_visibility_mat = np.zeros((cnt_gt, 100))
                
        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        flag = False     
        dummy_image = im_ipm.copy()
        
        dummy_image[:,:,:] = 0

        delta_z_dict = {}
        for i in range(cnt_gt):
            x_values = np.array(gt_lanes[i])[:, 0]
            z_values = np.array(gt_lanes[i])[:, 1]
            
            # TODO: remove this condition later
            if flag == True:
                x_ipm_values, y_ipm_values = transform_lane_g2gflat(gt_cam_height, x_values[:100], self.y_samples, z_values[:100])
                # remove those points with z_values > gt_cam_height, this is only for visualization on top-view
                x_ipm_values = x_ipm_values[np.where(z_values[:100] < gt_cam_height)]
                y_ipm_values = y_ipm_values[np.where(z_values[:100] < gt_cam_height)]

            else:
                x_ipm_values = x_values
                y_ipm_values = self.y_samples

            #for vis on IPM image
            x_ipm_values, y_ipm_values = homographic_transformation(self.H_g2ipm, x_ipm_values[:100], y_ipm_values)
            x_ipm_values = x_ipm_values.astype(np.int)
            y_ipm_values = y_ipm_values.astype(np.int)

            # draw on ipm
            for k in range(1, x_ipm_values.shape[0]):
                # only draw the visible portion
                if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k] and z_values[k] < gt_cam_height:

                    #TODO: Verify - or + for the value for delta_z
                    delta_z_dict.update({(x_ipm_values[k], y_ipm_values[k]): gt_cam_height + z_values[k]})
                    dummy_image = cv2.line(dummy_image, (x_ipm_values[k - 1], y_ipm_values[k - 1]),
                                    (x_ipm_values[k], y_ipm_values[k]), (color_list[i][0],0,0), 1)

        return dummy_image, delta_z_dict #(x,y: height ) (x -- 416, y --256)

class GeneratePertile():
    """
    The gt lanes are plotted in BEV and decimated into tiles of the repective size
    Helper function that will return the gt 
        regression targets: rho_ij, phi_ij 
        classification score per tile : c_ij

    return: gt_rho --> [batch_size, ipm_h/tile_size, ipm_w/tile_size]
            gt_phi --> [batch_size, n_bins, ipm_h/tile_size, ipm_w/tile_size]
            gt_c --> [batch_size, ipm_h/tile_size, ipm_w/tile_size]
    """
    def __init__(self, cfg):
        self.camera_intrinsics = cfg.K
        self.tile_size = cfg.tile_size
        #TODO: modify the top view region as per the GenLanenet paper
        self.top_view_region = cfg.top_view_region
        self.org_h = cfg.org_h
        self.org_w = cfg.org_w
        self.crop_y = cfg.crop_y
        self.ipm_w = cfg.ipm_w
        self.ipm_h = cfg.ipm_h
        self.n_bins = cfg.n_bins
        self.resize_h = cfg.resize_h
        self.resize_w = cfg.resize_w

        #CONDITION: There exist max 6 lanes in the dataset
        self.color_list = cfg.color_list
        self.calculate_bev_projection = CalculateDistanceAngleOffests(self.org_h, self.org_w, self.resize_h, self.resize_w, self.camera_intrinsics, self.ipm_w, self.ipm_h, self.crop_y, self.top_view_region)


    def generategt_pertile(self, gt_lanes, img , gt_cam_height, gt_cam_pitch):
        bev_projected_lanes, dz_dict = self.calculate_bev_projection.draw_bevlanes(gt_lanes, img,  gt_cam_height, gt_cam_pitch, self.color_list) ## returns an image array of gt lanes projected on BEV
        # cv2.imwrite("/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/datasets/complete_lines_test1001.jpg" , bev_projected_lanes)

        ##init gt arrays for rho, phi and classification score and delta_z
        grid_x = int(bev_projected_lanes.shape[0]/self.tile_size)
        grid_y = int(bev_projected_lanes.shape[1]/self.tile_size)

        gt_rho = np.zeros((grid_x, grid_y))
        gt_phi = np.zeros((self.n_bins, grid_x, grid_y))
        gt_c = np.zeros((grid_x, grid_y))
        gt_lane_class = np.zeros((grid_x, grid_y))
        gt_delta_z = np.zeros((grid_x, grid_y))
        
        bev_projected_lanes = bev_projected_lanes[:,:,0] #(416/16,256/16) --- (13,8)
        
        multiple = 1
        for r in range(0,bev_projected_lanes.shape[0],self.tile_size): 
            
            if r == 0:
                multiple = multiple
            else: 
                multiple = multiple + 2 

            for c in range(0,bev_projected_lanes.shape[1],self.tile_size):
                
                #calculating coordinates for o_prime with respect to origin at the center of each tile 
                x = -c-8
                y = -8 *multiple
    
                check_line_chords = np.argwhere(bev_projected_lanes[r:r+self.tile_size, c:c+self.tile_size] >0)
                tile_img = bev_projected_lanes[r:r+self.tile_size, c:c+self.tile_size]
                
                dz = []
                if r == 0 and c == 0:
                    #find delta_z from line_chords
                    for line_chord in check_line_chords:
                        try:
                            delta_z = dz_dict[(line_chord[0], line_chord[1])]
                        except:
                            continue
                        dz.append(delta_z)
                else: 
                    #find delta_z from line_chords
                    for line_chord in check_line_chords:
                        try:
                            delta_z = dz_dict[(line_chord[1]+c,line_chord[0]+r)]
                        except:
                            continue
                        dz.append(delta_z)
                
                if len(dz) != 0:    
                    mean_del_z  = sum(dz) / len(dz)
                else:
                    mean_del_z = 0

                accumulator, thetas, rhos, lane_exist = HoughLine(tile_img.copy())
                idx = np.argmax(accumulator)

                rho = rhos[int(idx / accumulator.shape[1])]
                # print(rho)
                theta = thetas[int(idx % accumulator.shape[1])] 
                
                ################### NEW IMPLEMENTATION OF RHO :: HESSE NORMAL FORM
                """
                r_prime = r - n.T * delta_o

                where n = [cos(theta)
                            sin(theta)
                            ]
                      delta_o = o_prime - o
                """
                theta = np.deg2rad(theta) #hough algo gives theta as degreees
                normal_vector = [ np.cos(theta), np.sin(theta) ]
                # normal_vector_normalized = normal_vector/ np.linalg.norm(normal_vector)
                
                o_prime = np.array([[-x],[-y]]) #TODO: verify the sign of x and y here
                delta_o = o_prime

                r_prime = rho - np.dot(normal_vector,delta_o)
                ###########################
            
                x_idx = int(r/self.tile_size)
                y_idx = int(c/self.tile_size)
                gt_rho[x_idx,y_idx] = r_prime
                
                phi_vec = binprob(self.n_bins, theta)
                # print(phi_vec, theta)
                # print("Checking the value of phi_vec in the generate function", phi_vec)
                gt_phi[:,x_idx,y_idx][:, np.newaxis] = phi_vec
                    
                if lane_exist == True:
                    gt_c[x_idx,y_idx] = 1
                    gt_delta_z[x_idx,y_idx] = mean_del_z
                else:
                    gt_c[x_idx,y_idx] = 0
                    gt_delta_z[x_idx,y_idx] = 0

                if 50 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 1
                elif 100 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 2 
                elif 150 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 3
                elif 200 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 4
                elif 250 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 5
                elif 255 in tile_img:
                    gt_lane_class[x_idx,y_idx] = 6
                else:
                    gt_lane_class[x_idx,y_idx] = 0
        return gt_rho, gt_phi, gt_c, gt_lane_class, gt_delta_z

def unnormalize(pred_data, min_pred, max_pred):
    """
    Unnormalize the predictions

    Args:
        pred_data: predictions
        min_pred: minimum value calculated from the training data
        max_pred: maximum value calculated from the training data
    
    Returns:
        unnormalized predictions
    """
    return (pred_data * (max_pred - min_pred)) + min_pred

class Apollo3d_loader(Dataset):
    def __init__(self, data_root, data_splits, phase = "train", cfg = None):
        super(Apollo3d_loader, self,).__init__()
        
        self.cfg = cfg
        self.data_root = data_root
        self.data_splits = data_splits
        self.phase = phase
        self.camera_intrinsics = self.cfg.K
        self.h_crop = self.cfg.crop_y
        self.generate_pertile = GeneratePertile(cfg)
        
        if phase == "train":
            self.data_filepath = os.path.join(self.data_splits, "train.json")
        elif phase == "test":
            self.data_filepath = os.path.join(self.data_splits, "test.json")
        if not os.path.exists(self.data_filepath):
            raise Exception('Fail to load the train.json file')
        
        self.load_dataset()
        
    def load_dataset(self):

        #loading data from the json file
        json_data = [json.loads(line) for line in open(self.data_filepath).readlines()]
        
        #extract image keys from the json file
        self.image_keys = [data['raw_file'] for data in json_data]
        
        """
        #just to ensure overfitting over one sample TODO: Add a condition for that
        """
        # self.image_keys = self.image_keys[0]

        self.data = {l['raw_file']: l for l in json_data}
    
    def normalize(self, data, min_data, max_data):
        """
        Normalise the data in the range [0,1]
        Args:
            data: data to be normalised
            min_data: minimum value of the training data (torch.tensor)
            max_data: maximum value of the training data (torch.tensor)

        return: normalised data 
        """
                
        norm_data = (data - min_data)/(max_data - min_data) 

        return norm_data 

    def __len__(self):
        return len(self.image_keys)
        # return 1

    def __getitem__(self, idx):
        batch = {}
        batch.update({"idx": idx})

        gtdata = self.data[self.image_keys[idx]]
        img_path = os.path.join(self.data_root, gtdata['raw_file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError('cannot find file: {}'.format(img_path))
        
        batch.update({"img_full_path": img_path})

        img = cv2.imread(img_path)

        gt_camera_height = np.array(gtdata['cam_height'])
        gt_camera_pitch =np.array(gtdata['cam_pitch'])

        batch.update({"gt_height":torch.from_numpy(gt_camera_height)})
        batch.update({"gt_pitch":torch.from_numpy(gt_camera_pitch)})
        
        gt_lanelines = gtdata['laneLines']
        batch.update({'gt_lanelines':gt_lanelines.copy()})

        gt_lateral_offset, gt_lateral_angleoffset, gt_cls_score, gt_lane_class, gt_delta_z = self.generate_pertile.generategt_pertile(gt_lanelines, img.copy(), gt_camera_height, gt_camera_pitch)

        norm_gt_lateral_offset = self.normalize(gt_lateral_offset, self.cfg.min_lateral_offset, self.cfg.max_lateral_offset)
        norm_gt_delta_z = self.normalize(gt_delta_z, self.cfg.min_delta_z, self.cfg.max_delta_z)
        
        if self.cfg.normalize == True:
            
            batch.update({'gt_rho':torch.from_numpy(norm_gt_lateral_offset)})    
            batch.update({'gt_delta_z':torch.from_numpy(norm_gt_delta_z)})
        else : 
            batch.update({'gt_rho':torch.from_numpy(gt_lateral_offset)})    
            batch.update({'gt_delta_z':torch.from_numpy(gt_delta_z)})
        
        batch.update({'gt_phi':torch.from_numpy(gt_lateral_angleoffset)})
        batch.update({'gt_clscore':torch.from_numpy(gt_cls_score)})
        batch.update({'gt_lane_class':torch.from_numpy(gt_lane_class)})

        #resize
        img = cv2.resize(img, (self.cfg.resize_w, self.cfg.resize_h), interpolation=cv2.INTER_AREA)
        
        if self.phase == "train": #training only
            img, aug_mat = data_aug_rotate(img)
            batch.update({"aug_mat": torch.from_numpy(aug_mat)})
        
        #convert the image to tensor
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #( BGR -> RGB)
        img = transforms.ToTensor()(img)
        
        img = transforms.Normalize(mean= self.cfg.img_mean, std=self.cfg.img_std)(img)
        batch.update({"image":img})
        print("checking the shape of the final image tensor::::", img.shape)
        return batch
        
def collate_fn(batch):
    """
    This function is used to collate the data for the dataloader
    """

    img_data = [item['image'] for item in batch]
    img_data = torch.stack(img_data, dim = 0)
    
    gt_camera_height_data = [item['gt_height'] for item in batch]
    gt_camera_height_data = torch.stack(gt_camera_height_data, dim = 0)

    gt_camera_pitch_data = [item['gt_pitch'] for item in batch]
    gt_camera_pitch_data = torch.stack(gt_camera_pitch_data, dim = 0)

    gt_lanelines_data = [item['gt_lanelines'] for item in batch]

    gt_rho_data = [item['gt_rho'] for item in batch]
    gt_rho_data = torch.stack(gt_rho_data, dim = 0)
    
    gt_phi_data = [item['gt_phi'] for item in batch]
    gt_phi_data = torch.stack(gt_phi_data, dim = 0)
    
    gt_cls_score_data = [item['gt_clscore'] for item in batch]
    gt_cls_score_data = torch.stack(gt_cls_score_data, dim = 0)

    gt_lane_class_data = [item['gt_lane_class'] for item in batch]
    gt_lane_class_data = torch.stack(gt_lane_class_data, dim = 0)

    gt_delta_z_data = [item['gt_delta_z'] for item in batch]
    gt_delta_z_data = torch.stack(gt_delta_z_data, dim = 0)
    
    gt_image_full_path =[item["img_full_path"] for item in batch]

    idx_data = [item['idx'] for item in batch]
    
    if 'aug_mat' in batch[0]:
        aug_mat_data = [item['aug_mat'] for item in batch]
        aug_mat_data = torch.stack(aug_mat_data, dim = 0)

        return [img_data, aug_mat_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data]
                    # 0      1                2                   3                    4                    5                 6                 7                 8          9              10                  11
    else: #no augmentation
        return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data]
                    #0      1                           2                   3                    4             5                 6                 7                 8          9                   10              
if __name__ == "__main__":

    #unit test for the data loader
    #TODO: add the hardcoded arguments to config file later on
    data_root = '/home/gjain2s/Documents/lane_detection_datasets/3d_dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/gjain2s/Documents/lane_detection_datasets/3d_dataset/data_splits/standard'
    config_path = '/home/gjain2s/Documents/E2E_3DLane_AuxNet/configs/3Dlane_detection/config_anchorless_3dlane.py'
    
    cfgs = Config.fromfile(config_path)
    vis = Visualization(cfgs.org_h, cfgs.org_w, cfgs.resize_h, cfgs.resize_w, cfgs.K, cfgs.ipm_w, cfgs.ipm_h, cfgs.crop_y, cfgs.top_view_region)
    dataset = Apollo3d_loader(data_root, data_splits, cfg = cfgs, phase = 'train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn, persistent_workers=True, prefetch_factor = cfgs.prefetch_factor)

    """
    Below code can be used to debug, verify and visualize the gt generating
    """
    start_point = time.time()
    for i, data in enumerate(loader):
        print(data[0].shape)
        # vis_rho_pred = data[5] #---> (b,13,8)
        # vis_delta_z_pred = data[9] #--> (b,13,8)
        # vis_cls_score_pred = data[7] # --> (b,13,8)
        # # print(vis_cls_score_pred)
        # vis_phi_pred = data[6] # --> (b,10,13,8) ---> (b,13,8)
        # vis_lane_class = data[8]#---> (b,13,8)
        
        # # print(vis_lane_class)
        
        # vis_cam_height = data[2].cpu().numpy()
        # vis_cam_pitch = data[3].cpu().numpy()
        
        
        # print(data[10])
        # # print("=======================> checking the shaoe of phi tensor", vis_phi_pred.shape)
        # # print(vis_rho_pred)


        # for b in range(vis_rho_pred.shape[0]):
        #     vis_img_path = data[10][b]

        # print("Sanity check if the correct image is used for gt", vis_img_path)
        # vis_img = cv2.imread(vis_img_path)
        
        # #offset predictions
        # vis_rho_pred_b = vis_rho_pred[b,:,:].detach().cpu().numpy()
        # vis_phi_pred_b = vis_phi_pred[b,:,:,:].detach().cpu().numpy()
        # vis_delta_z_pred_b = vis_delta_z_pred[b,:,:].detach().cpu().numpy()
        # vis_cls_score_pred_b = vis_cls_score_pred[b,:,:].detach().cpu().numpy()
        # print(vis_phi_pred_b.shape)

        # #unormalize the rho and delta z
        # if cfgs.normalize == True: ##TODO: off the normalize
        #     vis_rho_pred_b = vis_rho_pred_b * (cfgs.max_lateral_offset - cfgs.min_lateral_offset) + cfgs.min_lateral_offset
        #     vis_delta_z_pred_b = vis_delta_z_pred_b * (cfgs.max_delta_z - cfgs.min_delta_z) + cfgs.min_delta_z
        # else: 
        #     vis_rho_pred_b = vis_rho_pred_b
        #     vis_delta_z_pred_b = vis_delta_z_pred_b

        # vis_cam_height_b = vis_cam_height[b]
        # vis_cam_pitch_b = vis_cam_pitch[b]
        
        # clustered_tiles = vis_lane_class[b,:,:].cpu().numpy()
        
        # print("lane labels::",clustered_tiles)
        # print("check the values of rho", vis_rho_pred_b)
        
        # #extract points from predictions
        # points = [] ## ---> [[points lane1 (lists)], [points lane2(lists))], ...]
        # for i, lane_idx in enumerate(np.unique(clustered_tiles)): #must loop as the number of lanes present in the scene, max == 5 0 1,2,3, 4,
        #     print("========================")
        #     if lane_idx == 0: #no lane ::ignored
        #         continue
            
        #     curr_idx = np.where(clustered_tiles == lane_idx) # --> tuple (rows, comumns) idxs
        #     # print("lenght of the lane nidices", len(curr_idx[0]))
            
        #     ##
        #     rho_lane_i = vis_rho_pred_b[curr_idx[0], curr_idx[1]]
        #     # print("lenght of rho values or this points", len(rho_lane_i))
            
        #     phi_vec_lane_i =vis_phi_pred_b[:,curr_idx[0], curr_idx[1]] # ---> 1d array of 10 elements containing probs 
            
        #     # print("Checking the sahpe pf phi vector for one lane", phi_vec_lane_i.shape)
        #     # print("phi binned vectors per lane")
        #     for i in range(phi_vec_lane_i.shape[1]):
        #         print(phi_vec_lane_i[:,i])
            
        #     ##
        #     phi_lane_i = [palpha2alpha(phi_vec_lane_i[:,i]) for i in range(phi_vec_lane_i.shape[1])]
        #     print("phi values:::", phi_lane_i)

        #     ##
        #     delta_z_lane_i = vis_delta_z_pred_b[curr_idx[0], curr_idx[1]]
        #     print(delta_z_lane_i.shape)
        #     # print("check if the delta_z number thus points", len(delta_z_lane_i))
            
        #     # print("===============>",vis_cam_pitch_b)
        #     # print("===============>", vis_cam_height_b)
        
        #     """
        #     Here I will check all the shapes and make sure there is no problem
        #     """
        #     print("======================")
        #     print(len(phi_lane_i))
        #     print(delta_z_lane_i.shape)
        #     print(rho_lane_i.shape)
        #     print(vis_cam_pitch)
        #     # print(phi_lane_i.shape)

        #     ##
        #     points_lane_i = [polar_to_catesian(phi_lane_i[i], vis_cam_pitch_b, vis_cam_height_b, delta_z_lane_i[i], rho_lane_i[i]) for i in range(len(phi_lane_i))]
            
        #     # print("check the total number of points for a lane", len(points_lane_i))
        #     points.append(points_lane_i) 

        # print("Check the number of lanes ==============>", len(points))
        # fig1 = plt.figure(figsize=(10, 10))
        # ax2 = fig1.add_subplot(131, projection='3d')
        # ax1 = fig1.add_subplot(132)
        # for i in range(len(points)):
        #     x_values = np.array(points[i])[:, 0]
        #     y_values = np.array(points[i])[:, 1]
        #     z_values = np.array(points[i])[:, 2]

        #     ax2.plot(x_values, y_values, z_values, color = 'green', linewidth = 1 )
        #     ax1.plot(x_values, y_values)

        # pred_fig = mplfig_to_npimage(fig1)
        # cv2.imwrite("potter101.jpg", pred_fig)

        # # list containing arrays of lane points
        # # TODO: obtain a single plot with all the plots
        # # cv2.imwrite("vis_sanity_check_rgb.jpg", vis_img.copy())
        # # print(points)
        # pred_fig = vis.draw_lanes(points, vis_img.copy(), vis_cam_height_b, vis_cam_pitch_b) 
        # pred_fig = mplfig_to_npimage(pred_fig)
        # cv2.imwrite("vis_gt_sanity_check101.jpg", pred_fig)

        # pred_fig_real = vis.draw_lanes(data[4][b], vis_img.copy(), vis_cam_height_b, vis_cam_pitch_b)
        # pred_fig_real = mplfig_to_npimage(pred_fig_real)
        # cv2.imwrite("sanity_check_real_gt101.jpg", pred_fig_real)
        # assert 0