import numpy as np 
import torch
#TOOD: remove it later and verify it is of no use

dataset_dir = "/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/Apollo_Sim_3D_Lane_Release"
# #TODO: Optimize this config
# """
# Dataloader params
# """

# #TODO: Mya be the mean and std values are not correct for the dataset
# img_norm = dict(
#     mean=[103.939, 116.779, 123.68],
#     std=[1., 1., 1.]
# )

img_height = 360
img_width = 480
cut_height = 160
# ori_img_h = 720
# ori_img_w = 1280

# """
# training params
# """
# workers = 12
# num_classes = 6 + 1
num_classes = 1 + 1 # binary segmentation
# # ignore_label = 255

# test_json_file='/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple/test_label.json'

"""
2d model params
"""
featuremap_out_channel = 128
backbone = dict(
    type='ResNetWrapper',
    resnet_variant='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1],
    featuremap_out_channel = 128)

aggregator = dict(type= "SCNN")

heads = dict(type = 'PlainDecoder')

#TODO: move this to CLI args later
pretrained_2dmodel_path = "/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/nets/checkpoints/RGB_b16_r18_scnn_binary_2dLane_16_June_0.006351555231958628_90.pth"
lane_pred_dir = "/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/nets/3dlane_detection"

"""
3d model params (anchorless) for Apollo SIM3D dataset
"""
tile_size = 16
# encode_mode ="overlapping"
# encode_mode = "32x32non"
encode_mode = "1x1_mix_32X32"

#TODO: optimize a bit more later
#(channles, K_size, padding, stride)
if encode_mode == "overlapping":
    if tile_size == 32:
        encode_config = [(8,3,1,1), 'M', (16,3,1,1), 'M', (32,3,1,1), 'M', (64,3,1,1), "M", (64,3,1,1), "M", (64,3,1,1), (13,1,0,1)]
    elif tile_size == 16:
        encode_config = [(8,3,1,1), 'M', (16,3,1,1), 'M', (32,3,1,1), 'M', (64,3,1,1), "M", (64,3,1,1), (13,1,0,1)]

elif encode_mode == "32x32non":
    if tile_size == 32:
        encode_config = [(1,32,0,32), (8,1,0,1), (16,1,0,1), (13,1,0,1)]
    elif tile_size == 16:
        encode_config = [(1,16,0,16), (8,1,0,1), (16,1,0,1), (13,1,0,1)]

elif encode_mode == "1x1_mix_32X32":
    if tile_size == 32:
        encode_config = [(8,1,0,1), (16,1,0,1), (1,32,0,32), (13,1,0,1)]
    elif tile_size == 16:
        encode_config = [(8,1,0,1), (16,1,0,1), (1,16,0,16), (13,1,0,1)]

color_list = [[50,1],[100,2],[150,3],[200,4],[250,5],[255,6]] #(color, lane_class)
org_h = 1080
org_w = 1920
crop_y = 0
resize_h = 360 
resize_w = 480

ipm_h = 208 * 2
ipm_w = 128 * 2 

augmentation = True

# img_mean = [103.939, 116.779, 123.68]
# img_std = [1., 1., 1.]

img_mean = [0.485, 0.456, 0.406] 
img_std = [0.229, 0.224, 0.225]

no_centerline = True

"""
Below values are calculated by iterating over the train dataset
"""
min_lateral_offset = -23
max_lateral_offset = 23

#TODO: NOTE: need to calculate this value first over the train dataset
min_delta_z = -7.0678
max_delta_z = 3.5718

#init camera height and pitch
cam_height = 1.55
pitch = 3 #degrees

#camera instrinsics
K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

#TODO: change the top view region as per Genlanenet
# top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
batch_norm = True #bev encoder
embedding_dim = 5

##### Loss function params 
### regression and classification offsets and angles
n_bins = 10

# Discriminative Loss 

#TODO: Change the values as per the Gen_net paper
delta_pull = 0.1 ## delta_v 
delta_push = 3.0 ## delta_d 
embed_dim = 5

"""""
BIG BIG BIG NOTE: Is that the spatial size of ipm and BEV IPM projection are different?
Answer:: while cal gt the ipm_h and ipm_w are multiplied by a factor of 2. 
May be to process the masks for the section 3.2 I need not multiply the ipm_h and ipm_w by 2.
"""

# ###logging params
date_it = "9_July_"
train_run_name = date_it +  "Anchorless3DLane_norm_b8_YES0.001weights_16X1nonoverlap_CornFalse_0.001_0.1pullband_noclip_fixbranch_embed5" 
val_frequency = 500
vis_frequency = 100
train_log_frequency = 10

#if the predictions needs to be normalized
normalize = True
enable_clip = False
allign_corners = False
visualize_activations = True
fix_branch = True
weighted_loss = False

# #Hyperparams
epochs = 50
batch_size = 8
num_workers = 8
l2_lambda = 1e-4
lr = 0.001
lrs_cd = 0
lrs_factor = 0.75
lrs_min = 1e-6
lrs_patience = 3
lrs_thresh = 1e-4
prefetch_factor = 2
bg_weight = 0.4 #used in the loss function to reduce the importance of one class in tusimple

#TODO: try different combination later as per the gradients and in the end try to balance them
w_clustering_Loss = 1
w_classification_Loss = 0.1
threshold_score = 0.3

if enable_clip:
    grad_clip = 10
else: 
    grad_clip = torch.inf

fix_branch_epoch = 20

"""
NOTE:: NOTE:: NOTE:: NOTE::
In the end I would say that, yes this method of mine is one camera based. Just take the example of comma and decode
how can I make my approach work for any camera whatsoever, more over read that once paper which is claminig this also.

"""