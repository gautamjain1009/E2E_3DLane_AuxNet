"""
Dataloader params
"""
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 368
img_width = 640
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

dataset_path = '/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple'

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
        split='val',
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
training params
"""
workers = 12
num_classes = 6 + 1
# ignore_label = 255

test_json_file='/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple/test_label.json'

net = dict(
    type='ERFNet',
    num_classes = num_classes,
    exist_head = False # valid only for CUlane
)

###logging params
date_it = "20_March_" #TODO: remove it from argparse
train_run_name = "baseline_2dLane" + date_it
val_frequency = 1
train_log_frequency = 1

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
total_iter = (3616 // batch_size + 1) * epochs #TODO: change the number of iterations as per length of dataset
bg_weight = 0.4 #used in the loss function to reduce the importance of one class in tusimple