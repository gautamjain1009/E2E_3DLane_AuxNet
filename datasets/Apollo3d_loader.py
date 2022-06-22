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

            #for vis on original image
            x_2d, y_2d = projective_transformation(P_gt, x_values[:100], self.y_samples, z_values[:100])
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)

            #draw on 2d image
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k]:
                    #check if the point is in the image
                    if 0 <= x_2d[k] <= img.shape[1] and 0<= y_2d[k] <= img.shape[0] and  0 <= x_2d[k-1] <= img.shape[1] and 0 <= y_2d[k-1] <= img.shape[0]:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [255,0,0], 3)

            # draw on ipm
            for k in range(1, x_ipm_values.shape[0]):
                # print("first loop", vis)
                # only draw the visible portion
                if gt_visibility_mat[i, k - 1] and gt_visibility_mat[i, k] and z_values[k] < gt_cam_height:
                    # print("print here", vis)
                    #TODO: Verify - or + for the value for delta_z
                    dummy_image = cv2.line(dummy_image, (x_ipm_values[k - 1], y_ipm_values[k - 1]),
                                    (x_ipm_values[k], y_ipm_values[k]), (255,0,0), 1)

            #draw in 3d
            ax1.plot(x_values[np.where(gt_visibility_mat[i, :])],
                    self.y_samples[np.where(gt_visibility_mat[i, :])],
                    z_values[np.where(gt_visibility_mat[i, :])], color= 'green', linewidth=1)


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
        # print("Checking the shape of cropped image: ", img.shape)     

        # img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
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
        # cv2.imwrite("/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/complete_lines_test1001.jpg" , bev_projected_lanes)

        ##init gt arrays for rho, phi and classification score and delta_z
        grid_x = int(bev_projected_lanes.shape[0]/self.tile_size)
        grid_y = int(bev_projected_lanes.shape[1]/self.tile_size)

        gt_rho = np.zeros((grid_x, grid_y))
        gt_phi = np.zeros((self.n_bins, grid_x, grid_y))
        gt_c = np.zeros((grid_x, grid_y))
        gt_lane_class = np.zeros((grid_x, grid_y))
        gt_delta_z = np.zeros((grid_x, grid_y))
        
        bev_projected_lanes = bev_projected_lanes[:,:,0] #(416/32,256/32) --- (13,8)
        
        for r in range(0,bev_projected_lanes.shape[0],self.tile_size): ### r === 13 times
            for c in range(0,bev_projected_lanes.shape[1],self.tile_size): ### c == 8 times
                # cv2.imwrite(f"/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/test/img{r}_{c}.png",bev_projected_lanes[r:r+32, c:c+32])
                
                #TODO: replace 32 by self.tile_size
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

                accumulator, thetas, rhos, lane_exist = HoughLine(tile_img)
                idx = np.argmax(accumulator)

                rho = int(rhos[int(idx / accumulator.shape[1])])
                theta = thetas[int(idx % accumulator.shape[1])] #radians

                x_idx = int(r/self.tile_size)
                y_idx = int(c/self.tile_size)
                gt_rho[x_idx,y_idx] = rho
                phi_vec = binprob(self.n_bins, theta)
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

    def __getitem__(self, idx):
        batch = {}
        batch.update({"idx": idx})

        gtdata = self.data[self.image_keys[idx]]
        img_path = os.path.join(self.data_root, gtdata['raw_file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError('cannot find file: {}'.format(img_path))
        
        batch.update({"img_full_path": img_path})

        img = cv2.imread(img_path)
        
        #resize
        img = cv2.resize(img, (self.cfg.resize_h, self.cfg.resize_w), interpolation=cv2.INTER_AREA)

        if self.phase == "train": #training only
            img, aug_mat = data_aug_rotate(img)
            batch.update({"aug_mat": torch.from_numpy(aug_mat)})

        gt_camera_height = np.array(gtdata['cam_height'])
        gt_camera_pitch =np.array(gtdata['cam_pitch'])

        batch.update({"gt_height":torch.from_numpy(gt_camera_height)})
        batch.update({"gt_pitch":torch.from_numpy(gt_camera_pitch)})
        
        #TODO: correct the data representation of lane points while in the need of visulization
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
        
        #convert the image to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #( BGR -> RGB)
        img = transforms.ToTensor()(img)
        
        img = transforms.Normalize(mean= self.cfg.img_mean, std=self.cfg.img_std)(img)
        batch.update({"image":img})
        
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
    data_root = '/home/ims-robotics/Documents/gautam/dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/ims-robotics/Documents/gautam/dataset/data_splits/standard'
    config_path = '/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/configs/config_anchorless_3dlane.py'
    
    cfgs = Config.fromfile(config_path)

    dataset = Apollo3d_loader(data_root, data_splits, cfg = cfgs, phase = 'train')
    loader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, prefetch_factor=2, persistent_workers=True)

    max_list = []
    min_list = []

    start_point = time.time()
    for i, data in enumerate(loader):
        batch_time = time.time()
        
        print("time taken to process one batch",pprint_seconds(time.time() - batch_time))
        # print(data[9])
        
        
        start_point = batch_time
        # print("checking the detlat",data[9])
        # print("checking if class score",data[7])
        
    #     for j in range(cfgs.batch_size):
    #         print("checking min and max for iteration:::",i)
    #         max_list.append(data[9][j].max())
    #         min_list.append(data[9][j].min())


    # print("The max value of delta_z for the whole dataset is:::",  max(max_list))
    # print("The min value of delta_z for the whole dataset is:::",  min(min_list)) 