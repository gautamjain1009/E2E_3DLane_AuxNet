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

def deg2rad(angle):
    
    #convert angle from degree to radian
    a = (np.pi / 180) 
    b = angle * a

    return b 

def binprob(n_bin, angle, d2rad = False):
    
    """
    helper function to generate the bin prob vector
    (phi) from line angle for each tile
    
    """
    constant = 2*np.pi/n_bin
    
    if d2rad == True: 
        rad_angle = deg2rad(angle)
    else:
        rad_angle = angle

    bin_vector = np.zeros((n_bin, 1))
    
    for i in range(n_bin):
        p_aplha = 1 - (abs((constant * i) - rad_angle)/constant)
        bin_vector[i] = p_aplha

    bin_vector[bin_vector<0] = 0

    return bin_vector 

def HoughLine(image):
        ''' Basic Hough line transform that builds the accumulator array
        Input : image tile (gray scale image)
        Output : accumulator : the accumulator of hough space
                thetas : values of theta (0 : 360)
                rs : values of radius (-max distance : max distance)
        '''
        lane_exist = False

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
                    lane_exist = True
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
        return accumulator, thetas, rs, lane_exist

class CalculateDistanceAngleOffests(object):
    def __init__(self, org_h, org_w, K, ipm_w, ipm_h, crop_y, top_view_region):
        
        self.min_y = 0
        self.max_y = 80
        
        self.K = K
        
        self.crop_y = crop_y
        self.top_view_region = top_view_region

        """
            TODO: change the org_h and org_w according to feature map size in the network may be top view region too
        """
        self.resize_h = org_h
        self.resize_w = org_w

        # self.resize_h = resize_h
        # self.resize_w = resize_w
        
        self.ipm_w = 2*ipm_w
        self.ipm_h = 2*ipm_h
        
        #TODO: check the defination of these variables org_h, org_w, ipm_w, ipm_h etc. in terms of class variables

        self.H_crop = homography_crop_resize([org_h, org_w], self.crop_y, [self.resize_h, self.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(self.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)

        self.x_min = top_view_region[0, 0]
        self.x_max = top_view_region[1, 0]

        # self.y_samples = np.linspace(args.anchor_y_steps[0], args.anchor_y_steps[-1], num=100, endpoint=False)
        self.y_samples = np.linspace(self.min_y, self.max_y, num=100, endpoint=False)

    def draw_bevlanes(self, gt_lanes, img, gt_cam_height, gt_cam_pitch, color_list):
        
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)

        H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        # print("Checking the shape of original image: ", img.shape)

        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        # print("Checking the shape of cropped image: ", img.shape)     

        # img = img.astype(np.float) / 255
        # cv2.imwrite("/home/gautam/e2e/lane_detection/3d_approaches/Pytorch_Generalized_3D_Lane_Detection/tools/test1.jpg", img) #normalized
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
        # im_ipm = np.clip(im_ipm, 0, 1)
        # cv2.imwrite("/home/gautam/e2e/lane_detection/3d_approaches/Pytorch_Generalized_3D_Lane_Detection/tools/test_ipm.jpg", im_ipm)

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

        # cv2.imwrite("/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/viss_test.jpg", im_ipm)        
        dummy_image = im_ipm.copy()
        dummy_image[:,:,:] = 0
        
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
                    dummy_image = cv2.line(dummy_image, (x_ipm_values[k - 1], y_ipm_values[k - 1]),
                                        (x_ipm_values[k], y_ipm_values[k]), (color_list[i][0],0,0), 1)

        return dummy_image

#TODO: Rho values are all negative for some reason verify it is correct
def generategt_pertile(tile_size, gt_lanes, img , gt_cam_height, gt_cam_pitch):
    
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
    camera_intrinsics = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])
    
    #TODO: Add the harcoded params into the config file
    #TODO: modify the top view region
    top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    org_h = 1080
    org_w = 1920
    crop_y = 0

    ##MAIN TODO:::: check if ipm_w and ipm_h need to be upsampled or not (*2) or it should be same.
    ipm_w = 128
    ipm_h = 208
    n_bins = 10 

    #CONDITION: There exist max 6 lanes in the dataset
    color_list = [[50,1],[100,2],[150,3],[200,4],[250,5],[255,6]] #(color, lane_class) 

    #init the bev projection class
    calculate_bev_projection = CalculateDistanceAngleOffests(org_h, org_w, camera_intrinsics, ipm_w, ipm_h, crop_y, top_view_region)
    
    bev_projected_lanes = calculate_bev_projection.draw_bevlanes(gt_lanes, img , gt_cam_height, gt_cam_pitch, color_list) ## returns an image array of gt lanes projected on BEV

    # cv2.imwrite("/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/complete_lines_test.jpg" , bev_projected_lanes)

    ##init gt arrays for rho, phi and classification score
    gt_rho = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_phi = np.zeros((n_bins, int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_c = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))
    gt_lane_class = np.zeros((int(bev_projected_lanes.shape[0]/tile_size), int(bev_projected_lanes.shape[1]/tile_size)))

    bev_projected_lanes = bev_projected_lanes[:,:,0]
    for r in range(0,bev_projected_lanes.shape[0],tile_size): ### r === 13 times
        for c in range(0,bev_projected_lanes.shape[1],tile_size): ### c == 8 times
            # cv2.imwrite(f"/home/gautam/Thesis/E2E_3DLane_AuxNet/datasets/test/img{r}_{c}.png",bev_projected_lanes[r:r+32, c:c+32])
            tile_img = bev_projected_lanes[r:r+32, c:c+32]

            accumulator, thetas, rhos, lane_exist = HoughLine(tile_img)
            idx = np.argmax(accumulator)
            rho = int(rhos[int(idx / accumulator.shape[1])])
            theta = thetas[int(idx % accumulator.shape[1])] #radians

            x_idx = int(r/tile_size)
            y_idx = int(c/tile_size)
            gt_rho[x_idx,y_idx] = rho
    
            phi_vec = binprob(n_bins, theta)
            gt_phi[:,x_idx,y_idx][:, np.newaxis] = phi_vec

            if lane_exist == True:
                gt_c[x_idx,y_idx] = 1
            else:
                gt_c[x_idx,y_idx] = 0

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

    return gt_rho, gt_phi, gt_c, gt_lane_class


#TODO: add data augmentation as per the GeoNet paper
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
        
    def load_dataset(self):

        #TODO: Make it more efficient
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

        gt_camera_height = gtdata['cam_height'] 
        gt_camera_pitch =gtdata['cam_pitch']

        batch.update({"gt_height":gt_camera_height})
        batch.update({"gt_pitch":gt_camera_pitch})
        
        #TODO: correct the data representation of lane points while in the need of visulization
        gt_lanelines = gtdata['laneLines']
        batch.update({'gt_lanelines':gt_lanelines})

        gt_lateral_offset, gt_lateral_angleoffset, gt_cls_score, gt_lane_class = generategt_pertile(32, gt_lanelines, img, gt_camera_height, gt_camera_pitch)

        batch.update({'gt_rho':torch.from_numpy(gt_lateral_offset)})
        batch.update({'gt_phi':torch.from_numpy(gt_lateral_angleoffset)})
        batch.update({'gt_clscore':torch.from_numpy(gt_cls_score)})
        batch.update({'gt_lane_class':torch.from_numpy(gt_lane_class)})

        #TODO: add transforms normalize the image and BGR to RGB and update the other data accordingly
        
        #convert the image to tensor
        img = transforms.ToTensor()(img)
        img = img.float()
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

    return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data]

if __name__ == "__main__":

    #unit test for the data loader
    #TODO: add the hardcoded arguments to config file later on
    data_root = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/3D_Lane_Synthetic_Dataset/old_data_splits/standard'
    camera_intrinsics = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])
    top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    org_h = 1080
    org_w = 1920
    crop_y = 0


    dataset = Apollo3d_loader(camera_intrinsics, data_root, data_splits)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for i, data in enumerate(loader):
        print(data[0].shape)
        print(data[4].shape)
        print(data[5].shape)
        print(data[6].shape)
        print(data[7].shape)
        # print(data)
        # print(i)