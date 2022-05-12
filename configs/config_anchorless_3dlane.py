import numpy as np 

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
ori_img_h = 720
ori_img_w = 1280

# #for vis
# sample_y = range(710,150, -10)
# thr = 0.6

# train_augmentation = [
#     dict(type='RandomRotation'),
#     dict(type='RandomHorizontalFlip'),
#     dict(type='Resize', size=(img_width, img_height)),
#     dict(type='Normalize', img_norm=img_norm),
#     dict(type='ToTensor'),
# ] 

# val_augmentation = [
#     dict(type='Resize', size=(img_width, img_height)),
#     dict(type='Normalize', img_norm=img_norm),
#     dict(type='ToTensor')
# ] 

# dataset_path = '/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple'

# dataset = dict(
#     train=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='trainval',
#         transform = train_augmentation
#     ),
#     val=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='test',
#         transform = val_augmentation
#     ),
#     test=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='test',
#         transform = val_augmentation
#     )
# )

# """
# training params
# """
# workers = 12
num_classes = 6 + 1
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


"""
3d model params (anchorless) for Apollo SIM3D dataset
"""
color_list = [[50,1],[100,2],[150,3],[200,4],[250,5],[255,6]] #(color, lane_class)
org_h = 1080
org_w = 1920
crop_y = 0
resize_h = 360 
resize_w = 480

ipm_h = 208 * 2
ipm_w = 128 * 2 

augmentation = True
img_mean = [0.485, 0.456, 0.406] 
img_std = [0.229, 0.224, 0.225]

#init camera height and pitch
cam_height = 1.55
pitch = 3 #degrees

#camera instrinsics
K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

#TODO: change the top view region as per Genlanenet
top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
batch_norm = True #bev encoder
embedding_dim = 4 

##### Loss function params 
### regression and classification offsets and angles
n_bins = 10

# Discriminative Loss 

#TODO: Change the values as per the GEOnet paper
delta_pull = 0.5 ## delta_v 
delta_push = 3.0 ## delta_d 
tile_size = 32

"""""
BIG BIG BIG NOTE: Is that the spatial size of ipm and BEV IPM projection are different?
Answer:: while cal gt the ipm_h and ipm_w are multiplied by a factor of 2. 
May be to process the masks for the section 3.2 I need not multiply the ipm_h and ipm_w by 2.
"""

# ###logging params
# date_it = "25_March_"
# train_run_name = "Anchorless3DLane" + date_it
# val_frequency = 400
# train_log_frequency = 200

# #Hyperparams
# epochs = 100
batch_size = 1
num_workers = 1
# l2_lambda = 1e-4
# log_frequency_steps = 200
# lr = 0.001 
# lrs_cd = 0
# lrs_factor = 0.75
# lrs_min = 1e-6
# lrs_patience = 3
# lrs_thresh = 1e-4
# prefetch_factor = 2
# bg_weight = 0.4 #used in the loss function to reduce the importance of one class in tusimple