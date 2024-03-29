import sys
import time
sys.path.append("../")
import torch 
import numpy as np 
import cv2 
import os 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json 
from utils.helper_functions import *
import matplotlib.pyplot as plt
from torch import multiprocessing
from moviepy.video.io.bindings import mplfig_to_npimage
import subprocess
from utils.config import Config


def parse_affinity(cmd_output):
    ''' 
    Extracts the list of CPU ids from the `taskset -cp <pid>` command.

    example input : b"pid 38293's current affinity list: 0-3,96-99,108\n" 
    example output: [0,1,2,3,96,97,98,99,108]
    '''
    ranges_str = cmd_output.decode('utf8').split(': ')[-1].rstrip('\n').split(',')

    list_of_cpus = []
    for rng in ranges_str:
        is_range = '-' in rng

        if is_range:
            start, end = rng.split('-')
            rng_cpus = range(int(start), int(end)+1) # include end
            list_of_cpus += rng_cpus
        else:
            list_of_cpus.append(int(rng))

    return list_of_cpus


def set_affinity(pid, cpu_list):
    cmd = ['taskset', '-pc', ','.join(map(str, cpu_list)), str(pid)]
    subprocess.check_output(cmd, shell=False)


def get_affinity(pid):
    cmd = ['taskset', '-pc',  str(pid)]
    output = subprocess.check_output(cmd, shell=False)
    return parse_affinity(output)


def configure_worker(worker_id):
    '''
    Configures the worker to use the correct affinity.
    '''
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    worker_pid = os.getpid()
    dataset = worker_info.dataset
    avail_cpus = get_affinity(worker_pid)
    validation_term = 1 if dataset.validation else 0  # support two loaders at a time
    offset = len(avail_cpus) - (num_workers * 2) # keep the first few cpus free (it seemed they were faster, important for BackgroundGenerator)
    cpu_idx = max(offset + num_workers * validation_term + worker_id, 0)

    # force the process to only use 1 core instead of all
    set_affinity(worker_pid, [avail_cpus[cpu_idx]])


class BatchDataLoader:
    '''Assumes batch_size == num_workers to ensure same ordering of segments in each batch'''

    def __init__(self, loader, batch_size, mode):
        self.loader = loader
        self.batch_size = batch_size
        self.mode = mode

    def __iter__(self):
        bs = self.batch_size
        batch = [None] * bs
        current_bs = 0
        workers_seen = set()
        for d in self.loader:
            # print("checking gt lane score tensor", d[7])
            worker_id = d[-1]
            # print("checking if i reach at least until here==============", worker_id)

            # this means there're fewer segments left than the size of the batch — drop the last ones
            if worker_id in workers_seen:
                print(f'WARNING: sequence from worker:{worker_id} already seen in this batch. Dropping segments.')
                return  # FIXME: maybe pad the missing sequences with zeros and yield?

            batch[worker_id] = d
            current_bs += 1
            workers_seen.add(worker_id)

            if current_bs == bs:
                if self.mode == 'train':
                    collated_batch = self.collate_fn_train(batch)
                elif self.mode == 'test':
                    collated_batch = self.collate_fn_val(batch)

                # print(""checking the time)
                yield collated_batch
                batch = [None] * bs
                current_bs = 0
                workers_seen = set()

    def collate_fn_train(self, batch):

        img_data = [item[0] for item in batch]
        img_data = torch.stack(img_data, dim = 0)
        
        gt_camera_height_data = [item[2] for item in batch]
        gt_camera_height_data = torch.stack(gt_camera_height_data, dim = 0)

        gt_camera_pitch_data = [item[3] for item in batch]
        gt_camera_pitch_data = torch.stack(gt_camera_pitch_data, dim = 0)

        gt_lanelines_data = [item[4] for item in batch]

        gt_rho_data = [item[5] for item in batch]
        gt_rho_data = torch.stack(gt_rho_data, dim = 0)
        
        gt_phi_data = [item[6] for item in batch]
        gt_phi_data = torch.stack(gt_phi_data, dim = 0)
        
        gt_cls_score_data = [item[7] for item in batch]
        gt_cls_score_data = torch.stack(gt_cls_score_data, dim = 0)

        gt_lane_class_data = [item[8] for item in batch]
        gt_lane_class_data = torch.stack(gt_lane_class_data, dim = 0)

        gt_delta_z_data = [item[9] for item in batch]
        gt_delta_z_data = torch.stack(gt_delta_z_data, dim = 0)
        
        gt_image_full_path =[item[10] for item in batch]

        idx_data = [item[11] for item in batch]
        
        worker_idx = [item[12] for item in batch]

        aug_mat_data = [item[1] for item in batch]
        aug_mat_data = torch.stack(aug_mat_data, dim = 0)

        return [img_data, aug_mat_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data, worker_idx]
                    # 0      1                2                   3                    4                    5                 6                 7                 8          9              10                  11              12
        # else: #no augmentation
        #     return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data]
    
    def collate_fn_val(self, batch):
        img_data = [item[0] for item in batch]
        img_data = torch.stack(img_data, dim = 0)
        
        gt_camera_height_data = [item[1] for item in batch]
        gt_camera_height_data = torch.stack(gt_camera_height_data, dim = 0)

        gt_camera_pitch_data = [item[2] for item in batch]
        gt_camera_pitch_data = torch.stack(gt_camera_pitch_data, dim = 0)

        gt_lanelines_data = [item[3] for item in batch]

        gt_rho_data = [item[4] for item in batch]
        gt_rho_data = torch.stack(gt_rho_data, dim = 0)
        
        gt_phi_data = [item[5] for item in batch]
        gt_phi_data = torch.stack(gt_phi_data, dim = 0)
        
        gt_cls_score_data = [item[6] for item in batch]
        gt_cls_score_data = torch.stack(gt_cls_score_data, dim = 0)

        gt_lane_class_data = [item[7] for item in batch]
        gt_lane_class_data = torch.stack(gt_lane_class_data, dim = 0)

        gt_delta_z_data = [item[8] for item in batch]
        gt_delta_z_data = torch.stack(gt_delta_z_data, dim = 0)
        
        gt_image_full_path =[item[9] for item in batch]

        idx_data = [item[10] for item in batch]
        
        worker_idx = [item[11] for item in batch]
        
        return [img_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data, worker_idx]
                    # 0      1                          2                   3                    4             5                 6                 7                 8          9              10                  11              12

    def __len__(self):
        return len(self.loader)


class BackgroundGenerator(multiprocessing.Process):
    def __init__(self, generator):
        super(BackgroundGenerator, self).__init__() 
        # TODO: use prefetch factor instead of harcoded value
        self.queue = torch.multiprocessing.Queue(1)
        self.generator = generator
        self.start()

    def run(self):
        while True:
            for item in self.generator:
                # print(item.keys())

                # do not start (blocking) insertion into a full queue, just wait and then retry
                # this way we do not block the consumer process, allowing instant batch fetching for the model
                while self.queue.full():
                    time.sleep(2)

                self.queue.put(item)
            self.queue.put(None)

    def __iter__(self):
        try:
            next_item = self.queue.get()
            while next_item is not None:
                yield next_item
                next_item = self.queue.get()
        except (ConnectionResetError, ConnectionRefusedError) as err:
            print('[BackgroundGenerator] Error:', err)
            self.shutdown()
            raise StopIteration


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

        print("check if the number of lanes in visssss", cnt_gt)

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

        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))

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
        cv2.imwrite("v2investigate_if_rgb.jpg", img)
        bev_projected_lanes, dz_dict = self.calculate_bev_projection.draw_bevlanes(gt_lanes, img,  gt_cam_height, gt_cam_pitch, self.color_list) ## returns an image array of gt lanes projected on BEV
        cv2.imwrite("v2investigate_if.jpg" , bev_projected_lanes)

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
        print("print gt_lane_class gt inside the genrate function =====>", np.unique(gt_lane_class))
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

class Apollo3d_loader(IterableDataset):
    def __init__(self, data_root, data_splits, shuffle, phase, seed = 27, cfg = None):
        super(Apollo3d_loader, self,).__init__()
        
        self.cfg = cfg
        self.data_root = data_root
        self.data_splits = data_splits
        self.phase = phase
        self.camera_intrinsics = self.cfg.K
        self.h_crop = self.cfg.crop_y
        self.generate_pertile = GeneratePertile(cfg)
        self.validation = False
        self.seed = seed
        self.shuffle = shuffle
        
        if phase == "train":
            self.data_filepath = os.path.join(self.data_splits, "train.json")
        elif phase == "test":
            self.validation = True
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
        return len(self.image_keys)//self.cfg.batch_size

    def __iter__(self):

        #suffle datasubset after each epoch
        # if self.shuffle:
        #     rng = np.random.default_rng(self.seed)
        #     rng.shuffle(self.image_keys)

        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for idx in range(worker_id, len(self.image_keys), num_workers):

            batch = {}
            batch.update({"idx": idx})

            gtdata = self.data[self.image_keys[idx]]
            img_path = os.path.join(self.data_root, gtdata['raw_file'])
            
            if not os.path.exists(img_path):
                raise FileNotFoundError('cannot find file: {}'.format(img_path))
            
            batch.update({"img_full_path": img_path})

            print("check if the laoder is fetching the correct image_path", img_path)
            img = cv2.imread(img_path)
            
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

            #resize
            #TODO: NOTE: DISCLAIMER: to check if i shoudl send the full image for generating gt or this is okay
            img = cv2.resize(img, (self.cfg.resize_w, self.cfg.resize_h), interpolation=cv2.INTER_AREA)

            if self.phase == "train": #training only
                img, aug_mat = data_aug_rotate(img)
                batch.update({"aug_mat": torch.from_numpy(aug_mat)})
            #convert the image to tensor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #( BGR -> RGB)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean= self.cfg.img_mean, std=self.cfg.img_std)(img)
            batch.update({"image":img})
            
            batch.update({"worker_id": worker_id})

            # yield batch
        # [img_data, aug_mat_data, gt_camera_height_data, gt_camera_pitch_data, gt_lanelines_data, gt_rho_data, gt_phi_data, gt_cls_score_data, gt_lane_class_data, gt_delta_z_data, gt_image_full_path, idx_data]
            if self.phase == "train":
                yield batch["image"], batch["aug_mat"], batch["gt_height"], batch["gt_pitch"], batch["gt_lanelines"], batch["gt_rho"], batch["gt_phi"], batch["gt_clscore"], batch["gt_lane_class"], batch["gt_delta_z"], batch["img_full_path"], batch["idx"], batch["worker_id"]
                #           0           1                   2                   3                       4               5                   6               7                           8                   9                   10                      11              12              
            else: 
                yield batch["image"], batch["gt_height"], batch["gt_pitch"], batch["gt_lanelines"], batch["gt_rho"], batch["gt_phi"], batch["gt_clscore"], batch["gt_lane_class"], batch["gt_delta_z"], batch["img_full_path"], batch["idx"], batch["worker_id"]
                #           0           1                   2                   3                       4               5                   6               7                           8                   9                   10                      11              12              
if __name__ == "__main__":
    #unit test for the data loader
    #TODO: add the hardcoded arguments to config file later on
    data_root = '/home/ims-robotics/Documents/gautam/dataset/Apollo_Sim_3D_Lane_Release'
    data_splits = '/home/ims-robotics/Documents/gautam/dataset/data_splits/standard'
    config_path = '/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/configs/config_anchorless_3dlane.py'
    
    cfgs = Config.fromfile(config_path)
    vis = Visualization(cfgs.org_h, cfgs.org_w, cfgs.resize_h, cfgs.resize_w, cfgs.K, cfgs.ipm_w, cfgs.ipm_h, cfgs.crop_y, cfgs.top_view_region)

    dataset = Apollo3d_loader(data_root, data_splits, shuffle = False, cfg = cfgs, phase = 'train')
    loader = DataLoader(dataset, batch_size=None, num_workers=cfgs.batch_size, collate_fn= None, prefetch_factor=1, persistent_workers=True, worker_init_fn= configure_worker)
    
    loader = BatchDataLoader(loader, batch_size = cfgs.batch_size, mode = 'train')
    loader = BackgroundGenerator(loader)

    start_point = time.time()
    """
    Below code can be used to debug verify and visualize the gt generating
    """
    for i, data in enumerate(loader):
        print("begining data fetching ====>")
        print("check the bumber of iterations i fetch the batch:::::",i)
        vis_rho_pred = data[5] #---> (b,13,8)
        vis_delta_z_pred = data[9] #--> (b,13,8)
        vis_cls_score_pred = data[7] # --> (b,13,8)
        # print(vis_cls_score_pred)
        vis_phi_pred = data[6] # --> (b,10,13,8) ---> (b,13,8)
        vis_lane_class = data[8]
        print(vis_lane_class)
        vis_cam_height = data[2].cpu().numpy()
        vis_cam_pitch = data[3].cpu().numpy()
        
        print("===================len lanes gt",len(data[4][0]))
        print(data[10])

        for b in range(vis_rho_pred.shape[0]):
            vis_img_path = data[10][b]

            print("Sanity check if the correct image is used for gt", vis_img_path)
            vis_img = cv2.imread(vis_img_path)
            
            #offset predictions
            vis_rho_pred_b = vis_rho_pred[b,:,:].detach().cpu().numpy()
            vis_phi_pred_b = vis_phi_pred[b,:,:,:].detach().cpu().numpy()
            vis_delta_z_pred_b = vis_delta_z_pred[b,:,:].detach().cpu().numpy()
            vis_cls_score_pred_b = vis_cls_score_pred[b,:,:].detach().cpu().numpy()
            
            #unormalize the rho and delta z
            if cfgs.normalize == True: ##TODO: off the normalize
                vis_rho_pred_b = vis_rho_pred_b * (cfgs.max_lateral_offset - cfgs.min_lateral_offset) + cfgs.min_lateral_offset
                vis_delta_z_pred_b = vis_delta_z_pred_b * (cfgs.max_delta_z - cfgs.min_delta_z) + cfgs.min_delta_z
            else: 
                vis_rho_pred_b = vis_rho_pred_b
                vis_delta_z_pred_b = vis_delta_z_pred_b

            vis_cam_height_b = vis_cam_height[b]
            vis_cam_pitch_b = vis_cam_pitch[b]
            
            #Cluster the tile embedding as per lane class
            # return the tile labels: 0 marked as no lane
            # clustered_tiles = embedding_post_process(vis_embedding_b, vis_cls_score_pred_b) 
            ## replace this by lane_gt_cls tensor 
            clustered_tiles = vis_lane_class[b,:,:].cpu().numpy()
            
            print("check if the num of lanes::",clustered_tiles)
            
            #extract points from predictions
            points = [] ## ---> [[points lane1 (lists)], [points lane2(lists))], ...]
            for i, lane_idx in enumerate(np.unique(clustered_tiles)): #must loop as the number of lanes present in the scene, max == 5
                if lane_idx == 0: #no lane ::ignored
                    continue
                curr_idx = np.where(clustered_tiles == lane_idx) # --> tuple (rows, comumns) idxs
                # print("lenght of the lane nidices", len(curr_idx[0]))
                rho_lane_i = vis_rho_pred_b[curr_idx[0], curr_idx[1]]
                # print("lenght of rho values or this points", len(rho_lane_i))
                phi_vec_lane_i =vis_phi_pred_b[:,curr_idx[0], curr_idx[1]] # ---> 1d array of 10 elements containing probs 
                phi_lane_i = [palpha2alpha(phi_vec_lane_i[:,i]) for i in range(phi_vec_lane_i.shape[1])]
                # print("lenght of phi values thus points", len(phi_lane_i))
                delta_z_lane_i = vis_delta_z_pred_b[curr_idx[0], curr_idx[1]]
                # print("check if the delta_z number thus points", len(delta_z_lane_i))

                points_lane_i = [polar_to_catesian(phi_lane_i[i], vis_cam_pitch_b, vis_cam_height_b, delta_z_lane_i[i], rho_lane_i[i]) for i in range(len(phi_lane_i))]
                # print("check the total number of points for a lane", len(points_lane_i))
                points.append(points_lane_i) 

            print(len(points))
            print("====================================")
            print(len(data[4][0]))
            

            print("=============================")
            print("checking the number of points in gt lane points for first lane", len(data[4][b][0]))
            #list containing arrays of lane points
            #TODO: obtain a single plot with all the plots
            cv2.imwrite("v2sanity_check_rgb.jpg", vis_img.copy())
            pred_fig = vis.draw_lanes(points, vis_img.copy(), vis_cam_height_b, vis_cam_pitch_b)
            pred_fig = mplfig_to_npimage(pred_fig)
            cv2.imwrite("v2sanity_check.jpg", pred_fig)

            pred_fig_real = vis.draw_lanes(data[4][b], vis_img.copy(), vis_cam_height_b, vis_cam_pitch_b)
            pred_fig_real = mplfig_to_npimage(pred_fig_real)
            cv2.imwrite("v2sanity_check_real.jpg", pred_fig_real)


            break
        break

