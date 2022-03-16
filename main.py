import torch 
import cv2 

from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt

from models.baseline_model import ERFNet
from .timing import *
import logging 
logging.basicConfig(level = logging.DEBUG)

#build import for different modules
from datasets.registry import build_dataloader


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

    args = parser.parse_args()

    #parasing config file
    cfg = Config.fromfile(args.config)

    # building dataloader
    train_loader = build_dataloader(cfg.dataset.train, cfg, is_train = True)
    val_loadaer = build_dataloader(cfg.dataset.val, cfg, is_train = False)
    #TODO:
        #loss functions 
        # scheduler and optimizer 
        #trian loop /Eval loop
        #Integrate with wandb including model visulization 
        #idea for test loop and how the evaluation is carried out 
    

    #TODO: function
    if cfg.net.type == "baseline" and cfg.net.exist_head == False:
        model = ERFNet(cfg.net.num_classes)
    elif cfg.net.type == "baseline" and cfg.net.exist_head == True:
        #TODO: Solve the batch >1 for eexist head
        model = ERFNet(cfg.net.num_classes, cfg.net.exist_head)

    #segmentation loss
    criterion = torch.nn.NLLLoss()
    criterion_exist = torch.nn.BCEWithLogitsLoss()

    #optimizer and scheduler
    param_group = model.parameters()
    optimizer = topt.Adam(param_group, cfg.lr, weight_decay= cfg.l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor= cfg.lrs_factor, patience= cfg.lrs_patience,
                                                        threshold= cfg.lrs_thresh, verbose=True, min_lr= cfg.lrs_min,
                                                        cooldown=cfg.lrs_cd)
    #train loop 






    #eval Loop 















