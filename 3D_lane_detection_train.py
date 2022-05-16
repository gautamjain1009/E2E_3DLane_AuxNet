
from pprint import pprint
import torch 
import torch.nn as nn 
import cv2 
from tqdm import tqdm
import os 
# import dotenv
# dotenv.load_dotenv()
import wandb
import time

from utils.config import Config
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
from utils.helper_functions import *
from anchorless_detector import load_3d_model


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("=> Using '{}' for computation ".format(device))

    parser = argparse.ArgumentParser(description="Anchorless 3D Lane Detection Train")
    parser.add_argument("--config", type=str, default="configs/config_anchorless_3dlane.py", help="config file")
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--seed", type=int, default=27, help="random seed")
    parser.add_argument("--baseline", type=bool, default=False, help="enable baseline")

    #parsing args
    args = parser.parse_args()

    #load config file
    cfg = Config.fromfile(args.config)

    # for reproducibility
    torch.manual_seed(args.seed)

    # #trained model paths
    # checkpoints_dir = './nets/3dlane_detection/checkpoints'
    # result_model_dir = './nets/3dlane_detection/model_itr'
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # os.makedirs(result_model_dir, exist_ok=True)


    #dataloader


    #load model
 
    #TODO: Need to match the spatial size of the input image to the 2dmodel and thus the same goes for the 3d model 
    model2d = load_model(cfg, baseline = args.baseline).to(device)
    model3d = load_3d_model(cfg, device).to(device) 
    print(model3d)
    #optimizer and scheduler



    #loss functions



    #training loop
    #unit test
    a = torch.rand(1,3,360,480).to(device)

    o = model2d(a)
    print("checking the shape of the output tensor from 2d lane seg pipeline",o.shape)

    o = o.softmax(dim=1)
    o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
    o = o[:,1:2,:,:]    
    """
    #BIG BIG NOTE: In my case remember I need to obtain the binary seg masks for the other network output 
    currently I have 7 classes Investigate that when the network is finished
    """

    o2 = model3d(o)
    print("checking the shape of the output tensor",o2.shape)
