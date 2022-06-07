import numpy as np 
from scipy.interpolate import interp1d
import cv2 
import random
from sklearn.cluster import MeanShift, estimate_bandwidth

def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"
 
def HoughLine(image):
    
        '''  Hough line transform that builds the accumulator array from the center of the image.
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
        Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)/2)) #-- 23

        # Theta in range from -90 to 90 degrees
        thetas = np.deg2rad(np.arange(0, 360,1))
        
        #Range of radius
        # rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist )
        rs = np.arange(-Maxdist, Maxdist + 1, 1)

        #Create accumulator array and initialize to zero
        accumulator = np.zeros((len(rs), len(thetas)))
        y_idxs, x_idxs = np.nonzero(image)

        if len(x_idxs) == 0 or len(y_idxs) == 0:

            return accumulator, thetas, rs, lane_exist

        else:   
            lane_exist = True
            for i in range(len(x_idxs)):
                x = x_idxs[i]
                y = y_idxs[i]

                for j in range(len(thetas)):
                    rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + Maxdist)
                    try:
                        accumulator[rho, j] += 1
                    except:
                        continue

            return accumulator, thetas, rs, lane_exist

def embedding_post_process(embedding, bin_seg, band_width=1.5, max_num_lane=5):
    """
    First use mean shift to find dense cluster center.

    Arguments:
    ----------
    embedding: numpy [H, W, embed_dim]
    cls_score or bin_seg: both bin seg and lane_score represents the same thing 0 or 1 for that particular tile or pixel numpy [H, W]

    Return:
    ---------
    cluster_result: numpy [H, W], index of different lanes on each pixel
    """

    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)
    cluster_list = embedding[bin_seg>0]

    if len(cluster_list)==0:
        return cluster_result

    mean_shift = MeanShift(bandwidth=band_width, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(cluster_list)

    labels = mean_shift.labels_

    cluster_result[bin_seg>0] = labels + 1
    # print(np.unique(cluster_result))
    cluster_result[cluster_result > max_num_lane] = 0
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result==idx]) < 15:
        # print(len(cluster_result[cluster_result==idx]))
           cluster_result[cluster_result==idx] = 0
    return cluster_result

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

def homography_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def palpha2alpha(phi_vec):
    
    """
    Helper function to convert phi prob vector to abs angle value (0,360)

    return: phi angle in radians
    """
    
    phi_idx = np.argmax(phi_vec)

    alpha = (2 * np.pi / 10) * phi_idx
    return alpha 

def polar_to_catesian(pred_phi, cam_pitch, cam_height, delta_z_pred, rho_pred):
    """
    NOTE: this function is valid only for one tile
    convert the polar coordinates to cartesian coordinates
    :param pred_phi: predicted line angle
    :param cam_pitch: camera pitch (radians)
    :param cam_height: camera height (m)
    :param delta_z_pred: predicted delta z (m)
    :param rho_pred: predicted lateral offset
    :return:
    """
    # convert the polar coordinates to cartesian coordinates
    # theta = np.arctan(pred_phi)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(cam_pitch), np.sin(cam_pitch)],
                                [0, -np.sin(cam_pitch), np.cos(cam_pitch)]])
    translation_matrix = np.array([[rho_pred * np.cos(pred_phi)],[rho_pred * np.sin(pred_phi)], [delta_z_pred - cam_height]])
    
    cartesian_points = np.dot(rotation_matrix, translation_matrix) # --> (3, 1)

    cartesian_points = cartesian_points.ravel().tolist()

    return cartesian_points #--> List of 3 elements

# def polar_to_cartesian_sample():

#     pass  

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrixtransform_lane_g2gflat

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

def data_aug_rotate(img):
    
    rot = random.uniform(-np.pi/18, np.pi/18)
    # rot = random.uniform(-10, 10)
    center_x = img.shape[0]/ 2
    center_y = img.shape[1]/ 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.shape[0],img.shape[1]), flags=cv2.INTER_LINEAR)
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat

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
    :param resize_img_size: [resize_h, resize_w]0
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