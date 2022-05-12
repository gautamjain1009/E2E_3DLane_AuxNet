import sys
sys.path.append("../")
import torch 
import numpy as np 
import cv2 
import os 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging 
logging.basicConfig(level = logging.DEBUG)
import json 
from utils.helper_functions import *

"""
Import just for unit test
"""
from utils.config import Config

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

        ###TODO: Activate the resizing here
        # self.resize_h = resize_h
        # self.resize_w = resize_w
        
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

    def draw_bevlanes(self, gt_lanes, img, gt_cam_height, gt_cam_pitch, color_list):
        
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)

        H_g2im = homography_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        print("Checking the shape of original image: ", img.shape)

        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        # print("Checking the shape of cropped image: ", img.shape)     

        # img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
        # im_ipm = np.clip(im_ipm, 0, 1)
        cnt_gt = len(gt_lanes)
        gt_visibility_mat = np.zeros((cnt_gt, 100))
                
        # resample gt at y_samples
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

        return dummy_image, delta_z_dict

#TODO: Rho values are all negative for some reason verify it is correct
def generategt_pertile(gt_lanes, img , gt_cam_height, gt_cam_pitch, cfg):
    
    # n_tiles will be deifned as the number of tiles needed as per the size of the last feature map after bev encoder

    """
    The gt lanes are plotted in BEV and decimated into tiles of the repective size
    Helper function that will return the gt 
        regression targets: rho_ij, phi_ij 
        classification score per tile : c_ij

    return: gt_rho --> [batch_size, ipm_h/tile_size, ipm_w/tile_size]
            gt_phi --> [batch_size, n_bins, ipm_h/tile_size, ipm_w/tile_size]
            gt_c --> [batch_size, ipm_h/tile_size, ipm_w/tile_size]
    """
    #K
    camera_intrinsics = cfg.K
    tile_size = cfg.tile_size
    #TODO: Add the harcoded params into the config file
    
    #TODO: modify the top view region as per the GenLanenet paper
    top_view_region = cfg.top_view_region
    org_h = cfg.org_h
    org_w = cfg.org_w
    crop_y = cfg.crop_y

    ##MAIN TODO:::: check if ipm_w and ipm_h need to be upsampled or not (*2) or it should be same.
    ipm_w = cfg.ipm_w
    ipm_h = cfg.ipm_h
    n_bins = cfg.n_bins

    resize_h = cfg.resize_h
    resize_w = cfg.resize_w

    #CONDITION: There exist max 6 lanes in the dataset
    color_list = cfg.color_list

    #init the bev projection class
    calculate_bev_projection = CalculateDistanceAngleOffests(org_h, org_w, resize_h, resize_w, camera_intrinsics, ipm_w, ipm_h, crop_y, top_view_region)
    
    bev_projected_lanes, dz_dict = calculate_bev_projection.draw_bevlanes(gt_lanes, img,  gt_cam_height, gt_cam_pitch, color_list) ## returns an image array of gt lanes projected on BEV

    # cv2.imwrite("/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/complete_lines_test.jpg" , bev_projected_lanes)

    ##init gt arrays for rho, phi and classification score
    gt_rho = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_phi = np.zeros((n_bins, int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_c = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_lane_class = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_delta_z = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    
    bev_projected_lanes = bev_projected_lanes[:,:,0]
    for r in range(0,bev_projected_lanes.shape[0],tile_size): ### r === 13 times
        for c in range(0,bev_projected_lanes.shape[1],tile_size): ### c == 8 times
            # cv2.imwrite(f"/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/test/img{r}_{c}.png",bev_projected_lanes[r:r+32, c:c+32])
            
            #TODO: replace 32 by tile_size
            check_line_chords = np.argwhere(bev_projected_lanes[r:r+32, c:c+32] >0) ## denotes a patch of size 32x32
            tile_img = bev_projected_lanes[r:r+32, c:c+32]
            
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

            #TODO:When RHO: -45(no lane) and RHO: != 45 (lane exist), verify If i need to train with -45 or 0
            rho = int(rhos[int(idx / accumulator.shape[1])])
            theta = thetas[int(idx % accumulator.shape[1])] #radians

            x_idx = int(r/tile_size)
            y_idx = int(c/tile_size)
            gt_rho[x_idx,y_idx] = rho
    
            phi_vec = binprob(n_bins, theta)
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

class Apollo3d_loader(Dataset):
    def __init__(self, data_root, data_splits, phase = "train", cfg = None):
        super(Apollo3d_loader, self,).__init__()
        
        self.cfg = cfg
        self.data_root = data_root
        self.data_root = data_root
        self.data_splits = data_splits
        self.phase = phase
        self.camera_intrinsics = self.cfg.K
        self.h_crop = self.cfg.crop_y
        
        if phase == "train":
            self.data_filepath = os.path.join(self.data_splits, "train.json")
        else :
            self.data_filepath = os.path.join(self.data_splits, "test.json")
        
        self.load_dataset()
        
    def load_dataset(self):

        if not os.path.exists(self.data_filepath):
            raise Exception('Fail to load the train.json file')

        #loading data from the json file
        json_data = [json.loads(line) for line in open(self.data_filepath).readlines()]
        
        #extract image keys from the json file
        self.image_keys = [data['raw_file'] for data in json_data]

        self.data = {l['raw_file']: l for l in json_data}
    
    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):

        batch = {}

        gtdata = self.data[self.image_keys[idx]]
        img_path = os.path.join(self.data_root, gtdata['raw_file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError('cannot find file: {}'.format(img_path))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #crop
        img = img[:self.cfg.org_h - self.h_crop, :self.cfg.org_w]
        #resize
        img = cv2.resize(img, (self.cfg.resize_h, self.cfg.resize_w), interpolation=cv2.INTER_AREA)

        if self.cfg.augmentation:
            img, aug_mat = data_aug_rotate(img)
            batch.update({"aug_mat": torch.from_numpy(aug_mat)})

        gt_camera_height = gtdata['cam_height'] 
        gt_camera_pitch =gtdata['cam_pitch']

        batch.update({"gt_height":gt_camera_height})
        batch.update({"gt_pitch":gt_camera_pitch})
        
        #TODO: correct the data representation of lane points while in the need of visulization
        gt_lanelines = gtdata['laneLines']
        batch.update({'gt_lanelines':gt_lanelines})

        gt_lateral_offset, gt_lateral_angleoffset, gt_cls_score, gt_lane_class, gt_delta_z = generategt_pertile(gt_lanelines, img, gt_camera_height, gt_camera_pitch, self.cfg)

        batch.update({'gt_rho':torch.from_numpy(gt_lateral_offset)})
        batch.update({'gt_phi':torch.from_numpy(gt_lateral_angleoffset)})
        batch.update({'gt_clscore':torch.from_numpy(gt_cls_score)})
        batch.update({'gt_lane_class':torch.from_numpy(gt_lane_class)})
        batch.update({'gt_delta_z':torch.from_numpy(gt_delta_z)})
        
        #convert the image to tensor
        img = transforms.ToTensor()(img)
        img = img.float()
        img = transforms.Normalize(mean= self.cfg.img_mean, std=self.cfg.img_std)(img)
 
        batch.update({"image":img})

        return batch
        
def collate_fn(batch):
    """
    This function is used to collate the data for the dataloader
    """

    img_data = [item['image'] for item in batch]
    img_data = torch.stack(img_data, dim = 0)
    
    #TODO: stack them in the form of tensors if needed
    gt_camera_height_data = [item['gt_height'] for item in batch]
    gt_camera_pitch_data = [item['gt_pitch'] for item in batch]
    gt_lanelines_data = [item['gt_lanelines'] for item in batch] #need to check the data representation of lanelines (vis)
    
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

    if 'aug_mat' in batch[0]:
        aug_mat_data = [item['aug_mat'] for item in batch]
        aug_mat_data = torch.stack(aug_mat_data, dim = 0)

        return [img_data, aug_mat_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data]
                    # 0      1                2                   3                    4                    5                 6                 7                 8          9
    else: #no augmentation
        return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data]

if __name__ == "__main__":

    #unit test for the data loader
    #TODO: add the hardcoded arguments to config file later on
    data_root = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/3D_Lane_Synthetic_Dataset/old_data_splits/standard'
    config_path = '/home/gautam/Thesis/E2E_3DLane_AuxNet/configs/config_anchorless_3dlane.py'
    
    # top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    # org_h = 1080
    # org_w = 1920
    # crop_y = 0

    cfgs = Config.fromfile(config_path)

    dataset = Apollo3d_loader(data_root, data_splits, cfg = cfgs)
    loader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, num_workers=cfgs.num_workers, collate_fn=collate_fn)
    
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for i, data in enumerate(loader):
        print("Checking the shape of the image",data[0].shape)
        print("Checking the shape of the aug_mat",data[1].shape)
        print("Checking the values of RHO",data[5])
        print("Checking the values of Phi vector",data[6])
        print("Checking teh values of cls_score",data[7])
        print("Checking the values of lane class",data[8])
        print("Checking the values of delta_z",data[9])