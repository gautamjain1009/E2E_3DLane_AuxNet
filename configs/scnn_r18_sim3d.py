import numpy as np 
import os 
"""
Dataloader params
"""


vgg_mean = np.array([0.485, 0.456, 0.406])
vgg_std = np.array([0.229, 0.224, 0.225])

resize_h = 360
resize_w = 480
img_height = 360
img_width = 480
crop_y = 0

#TODO: remove this duplicay(required by model and visualization while training)
org_h = 1080
org_w = 1920
ori_img_h = 1080
ori_img_w = 1920
cut_height = 0

#for vis
sample_y = range(710,150, -10)
thr = 0.6

K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])
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

dataset_path = '/home/ims-robotics/Documents/gautam/dataset/Apollo_Sim_3D_Lane_Release'

data_split_path = "/home/ims-robotics/Documents/gautam/dataset/data_splits" 
type_split = "standard"

if type_split == "standard":
    train_split = type_split + "/train.json"
    train_file = os.path.join(data_split_path,train_split)

    val_split = type_split + "/test.json"
    val_file = os.path.join(data_split_path,val_split ) 
# elif type_split == "rare_subset":
#     train_file = os.path.join()
#     val_file = os.path.join() 
# elif type_split == "illus_chg":
#     train_file = os.path.join()
#     val_file = os.path.join()


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

"""
model params
"""
featuremap_out_channel = 128
featuremap_out_stride = 8

backbone = dict(
    type='ResNetWrapper',
    resnet_variant='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    featuremap_out_channel = 128)

aggregator = dict(type= "RESA",
                direction=['d', 'u', 'r', 'l'],
                alpha=2.0,
                iter=4,
                conv_stride=9)

heads = dict(type = 'BUSD')

"""
training params
"""
workers = 8
num_class = 1 + 1
num_classes = 2
# ignore_label = 255

test_json_file='/home/ims-robotics/Documents/gautam/dataset/tusimple/test_label.json'

###logging params
date_it = "_8Aug_March_"
train_run_name = "RESA_res18_sim3d_b8" + date_it
val_frequency = 400
train_log_frequency = 20

#Hyperparams
epochs = 100
batch_size = 8 
l2_lambda = 1e-4
log_frequency_steps = 200
lr = 0.001 
lrs_cd = 0
lrs_factor = 0.75
lrs_min = 1e-6
lrs_patience = 3
lrs_thresh = 1e-4
prefetch_factor = 2
bg_weight = 0.4 #used in the loss function to reduce the importance of one class in tusimple


train_type = "binary"