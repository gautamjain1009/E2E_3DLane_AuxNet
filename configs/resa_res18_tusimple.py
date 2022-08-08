"""
Dataloader params
"""

img_norm = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

img_height = 360
img_width = 480
cut_height = 160
ori_img_h = 720
ori_img_w = 1280

#for vis
sample_y = range(710,150, -10)
thr = 0.6

train_augmentation = [
    dict(type='RandomRotation'),
    dict(type='RandomHorizontalFlip'),
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor'),
] 

val_augmentation = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor')
] 

dataset_path = '/home/ims-robotics/Documents/gautam/dataset/tusimple'

dataset = dict(
    train=dict(
        type='TusimpleLoader',
        data_root=dataset_path,
        split='trainval',
        transform = train_augmentation
    ),
    val=dict(
        type='TusimpleLoader',
        data_root=dataset_path,
        split='test',
        transform = val_augmentation
    ),
    test=dict(
        type='TusimpleLoader',
        data_root=dataset_path,
        split='test',
        transform = val_augmentation
    )
)

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
num_classes = 1 + 1
# ignore_label = 255

test_json_file='/home/ims-robotics/Documents/gautam/dataset/tusimple/test_label.json'

###logging params
date_it = "_25_March_"
train_run_name = "RESA_res18_tusimple_b8" + date_it
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