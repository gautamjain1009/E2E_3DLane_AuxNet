import os
import os.path as ops
import numpy as np
import cv2
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F


"""
The below code is modified from: https://github.com/yuliangguo/Codes-for-Lane-Detection/blob/sim_data_adaption/ERFNet-CULane-PyTorch/dataset/sim3d_dataset.py
"""
class LaneDataset(Dataset):
    def __init__(self, args, dataset_path, json_file_path, transform=None, data_aug=False):
        print(args.num_class)
        self.is_testing = ('test' in json_file_path) # 'val'
        self.num_class = args.num_class

        # define image pre-processor
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)
        self.data_aug = data_aug

        # dataset parameters
        self.img_path = dataset_path
        self.transform = transform
        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y
        self.K = args.K

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

        self._label_image_path, self._label_laneline_all,\
            self._gt_cam_height_all, self._gt_cam_pitch_all,\
            self._gt_class_label_all = self.init_dataset_3D(dataset_path, json_file_path)

    def __len__(self):
        return len(self._label_image_path)

    def __getitem__(self, idx):
        img_name = self._label_image_path[idx]
        cam_pitch = self._gt_cam_pitch_all[idx]
        cam_height = self._gt_cam_height_all[idx]

        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=Image.BILINEAR)
        if self.data_aug:
            img_rot, aug_mat = data_aug_rotate(image)
            image = Image.fromarray(img_rot)
        image = self.totensor(image).float()
        image = self.normalize(image)

        # prepare binary segmentation label map
        label_map = np.zeros((self.h_net, self.w_net), dtype=np.int8)
        gt_lanes = self._label_laneline_all[idx]
        gt_labels = self._gt_class_label_all[idx]
        for i, lane in enumerate(gt_lanes):
            # skip the class label beyond consideration
            if gt_labels[i] <= 0 or gt_labels[i] > self.num_class:
                continue

            P_g2im = projection_g2im(cam_pitch, cam_height, self.K)
            M = np.matmul(self.H_crop, P_g2im)

            # update transformation with image augmentation
            if self.data_aug:
                M = np.matmul(aug_mat, M)
            x_2d, y_2d = projective_transformation(M, lane[:, 0],
                                                   lane[:, 1], lane[:, 2])
            for j in range(len(x_2d) - 1):
                label_map = cv2.line(label_map,
                                 (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j+1]), int(y_2d[j+1])),
                                 color=np.asscalar(gt_labels[i]), thickness=3)
        
        label_map = self.binary_segmask(label_map)
        label_map = torch.from_numpy(label_map.astype(np.int32)).contiguous().long()

        # if self.transform:
        #     image, label = self.transform((image, label))
        #     image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        #     label = torch.from_numpy(label).contiguous().long()

        batch = {} 
        batch.update({'img':image})
        batch.update({'binary_mask':label_map})
        batch.update({'full_img_path':img_name})
        # return image, label_map
        return batch

    def binary_segmask(self,mask_i):    
        #assuming there are at max 6 lanes in the dataset

        mask_i[mask_i ==1] = 1
        mask_i[mask_i ==2] = 1
        mask_i[mask_i ==3] = 1
        mask_i[mask_i ==4] = 1
        mask_i[mask_i ==5] = 1
        mask_i[mask_i ==6] = 1
    
        return mask_i 

    def init_dataset_3D(self, dataset_base_dir, json_file_path):
        """
        :param dataset_info_file:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []
        gt_cam_height_all = []
        gt_cam_pitch_all = []
        gt_class_label_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)

                image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                gt_lane_pts = info_dict['laneLines']
                gt_lane_visibility = info_dict['laneLines_visibility']
                if len(gt_lane_pts) == 0:
                    continue

                label_image_path.append(image_path)
                gt_starting_x = []
                for i, lane in enumerate(gt_lane_pts):
                    # A GT lane can be either 2D or 3D
                    # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                    lane = np.array(lane)
                    gt_lane_pts[i] = lane
                    gt_lane_visibility[i] = np.array(gt_lane_visibility[i])
                    gt_starting_x.append(lane[0, 0])
                gt_lane_pts = [prune_3d_lane_by_visibility(gt_lane, gt_lane_visibility[k]) for k, gt_lane in
                               enumerate(gt_lane_pts)]

                # determine gt laneline's ego id by its starting x position
                gt_starting_x = np.array(gt_starting_x)
                sort_id = np.argsort(gt_starting_x)
                gt_starting_x = gt_starting_x[sort_id]
                first_pos_idx = np.where(gt_starting_x > 0)[0][0]
                class_diff = self.num_class/2+1-first_pos_idx
                gt_class = np.array([idx for idx in range(gt_starting_x.shape[0])]) + class_diff
                gt_class_label_all.append(gt_class)

                gt_lane_pts = [gt_lane_pts[ind] for ind in sort_id]
                gt_laneline_pts_all.append(gt_lane_pts)

                gt_cam_height = info_dict['cam_height']
                gt_cam_height_all.append(gt_cam_height)
                gt_cam_pitch = info_dict['cam_pitch']
                gt_cam_pitch_all.append(gt_cam_pitch)

        label_image_path = np.array(label_image_path)
        gt_cam_height_all = np.array(gt_cam_height_all)
        gt_cam_pitch_all = np.array(gt_cam_pitch_all)

        return label_image_path, gt_laneline_pts_all, gt_cam_height_all, gt_cam_pitch_all, gt_class_label_all


def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im


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


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d


def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    # rot = random.uniform(-10, 10)
    center_x = img.width / 2
    center_y = img.height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    # img_rot = img.rotate(rot)
    # rot = rot / 180 * np.pi
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat

