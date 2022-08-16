#TODO: Change the name of the script to 2D_lane_detection_train.py

from contextlib import redirect_stderr
from pprint import pprint
import torch 
import torch.nn as nn
import cv2 
from tqdm import tqdm
import os 
import dotenv
dotenv.load_dotenv()
import wandb
import time
import sys

from utils.config import Config
from utils.visualization import LaneVisualisation
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt

from timing import * 
import logging 
logging.basicConfig(level = logging.DEBUG)

#build import for different modules
from datasets.registry import build_dataloader
from models.build_model import load_model

#import for sim3d loader 
from datasets.sim3d_binseg_loader import LaneDataset
from utils.segmentation_loss import DiceLoss, FocalLoss, FastConfusionMatrix


from models import baseline_model

def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"

def one_hot_convert(label, device):
    batch_size = label.shape[0]
    n_classes = 2
    h, w = label.shape[1], label.shape[2]

    one_hot_label = label.clone()

    one_hot_label = one_hot_label.unsqueeze(1)
    one_hot = torch.zeros(batch_size, n_classes, h, w).to(device)

    one_hot.scatter_(1, one_hot_label, 1)
    return one_hot 

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        # TODO: why the trained model do not have modules in name?
        if name[7:] not in list(own_state.keys()) or 'output_conv' in name:
            ckpt_name.append(name)
            # continue
        own_state[name[7:]].copy_(param)
        cnt += 1
    print('#reused param: {}'.format(cnt))
    return model
    
if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=>Using '{}' for computation.".format(device))

    parser = argparse.ArgumentParser(description='2D Lane detection')
    parser.add_argument('--config', help = 'path of train config file')
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--seed", type=int, default=27, help="random seed")
    parser.add_argument("--baseline", type=bool, default=False, help="enable baseline")
    parser.add_argument("--dataset", type=str, default="culane-tusimple", help="dataset used for training")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="loss function used for training")

    #parsing args
    args = parser.parse_args()

    #parasing config file
    cfg = Config.fromfile(args.config)
    
    #init vis class
    vis = LaneVisualisation(cfg)

    #wandb init
    run = wandb.init(entity = os.environ["WANDB_ENTITY"], project = os.environ["WANDB_PROJECT"], name = cfg.train_run_name, mode = 'offline' if args.no_wandb else 'online') 
    
    # for reproducibility
    torch.manual_seed(args.seed)

    #trained model paths
    checkpoints_dir = './nets/checkpoints/' + cfg.train_run_name
    result_model_dir = './nets/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    if args.dataset == "culane-tusimple":
        # dataloader for tusimple and culane
        print("Using TuSimple or CUlane Dataset ================>")
        train_loader = build_dataloader(cfg.dataset.train, cfg, is_train = True)
        val_loader = build_dataloader(cfg.dataset.val, cfg, is_train = False)
    
    elif args.dataset == 'sim3d':
        #dataloader for Apollo sim 3d dataset 
        print("using Apollo Sim3D dataset ====================>")
        train_dataset = LaneDataset(cfg, cfg.dataset_path, cfg.train_file, data_aug=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                           shuffle=True, num_workers=cfg.workers, pin_memory=False, drop_last=True)

        val_dataset = LaneDataset(cfg, cfg.dataset_path, cfg.val_file, data_aug=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                                shuffle=False, num_workers=cfg.workers, pin_memory=False)
        val_loader.is_testing = True

    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)
    
    print("===> batches in train loader", train_loader_len)
    print("===> batches in val loader", val_loader_len)
    
    
    model = baseline_model.ERFNet(2)
    # print(model)
    # print("model successfully loaded")
    checkpoint = torch.load("/home/ims-robotics/Documents/gautam/Pytorch_Generalized_3D_Lane_Detection_fork/pretrained/erfnet_model_sim3d.tar")
    
    model = load_my_state_dict(model, checkpoint['state_dict'])
    
    print("loading pretrained weights for ERFNet")

    
    ### Loading for custom models
    # model = load_model(cfg, baseline = args.baseline)
    model = model.to(device)
    
    wandb.watch(model)

    confusion_matrix = FastConfusionMatrix(num_classes=2)
    model.eval()         
    print(">>>>>>>>>Validating<<<<<<<<<")
    
    with torch.no_grad():
        val_pred = []
        pred_out = {}
        Iou_batch_list = []

        for val_itr, val_data in enumerate(val_loader):
            
            val_gt_mask = val_data['binary_mask'].to(device).long()
            val_img = val_data['img'].to(device)
            
            val_seg_out = model(val_img)

                #calcualte IOU
            metric_batch = iou(torch.argmax(val_seg_out,1), val_gt_mask)
            print(metric_batch)
            if isinstance(metric_batch, float) :
                continue
            else: 
                Iou_batch_list.append(metric_batch)
            confusion_matrix.update(one_hot_convert(val_gt_mask, device).flatten(start_dim=1), val_seg_out.flatten(start_dim=1))
            
        val_data_IOU = torch.mean(torch.stack(Iou_batch_list))
        IOU, mean_iou = confusion_matrix.get_metrics()

        print("The correct calculation of IoU is:", IOU)
        print("My old version of IOU is:" , val_data_IOU)  
   