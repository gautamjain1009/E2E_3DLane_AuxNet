import torch 
import cv2 
import tqdm 
import os 
from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt

# from models.baseline_model import ERFNet
# from .timing import *
import logging 
logging.basicConfig(level = logging.DEBUG)

#build import for different modules
from datasets.registry import build_dataloader
from models.registry import build_baseline

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

    #trained model paths
    checkpoints_dir = './nets/checkpoints'
    result_model_dir = './nets/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    # dataloader
    train_loader = build_dataloader(cfg.dataset.train, cfg, is_train = True)
    val_loadaer = build_dataloader(cfg.dataset.val, cfg, is_train = False)
    
    #TODO:
        #loss functions 
        # scheduler and optimizer 
        #trian loop /Eval loop
        #Integrate with wandb including model visulization 
        #idea for test loop and how the evaluation is carried out 
        #checkpoint dirs

    """ model defination TODO: add to build function"""
    def reinitialize_weights(layer_weight):
        torch.nn.init.xavier_uniform_(layer_weight)

    model = build_baseline(cfg.net, cfg)
    model = model.to(device)
           
    print("before",model.lane_exist.linear1.bias)
    
    #reinitialize model weights
    for name, layer in model.named_modules():
        
        if isinstance(layer,torch.nn.Conv2d):
            reinitialize_weights(layer.weight)
            try: 
                layer.bias.data.fill_(0.01)  
            except: ## TODO: Verify one layers bias is NONE? 
                pass 
            
        
        elif isinstance(layer, torch.nn.Linear):
            reinitialize_weights(layer.weight)
            layer.bias.data.fill_(0.01)

    print("After", model.lane_exist.linear1.bias)
        

    #segmentation loss
    criterion = torch.nn.NLLLoss().to(device)
    criterion_exist = torch.nn.BCEWithLogitsLoss().to(device)

    #optimizer and scheduler
    param_group = model.parameters()
    optimizer = topt.Adam(param_group, cfg.lr, weight_decay= cfg.l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor= cfg.lrs_factor, patience= cfg.lrs_patience,
                                                        threshold= cfg.lrs_thresh, verbose=True, min_lr= cfg.lrs_min,
                                                        cooldown=cfg.lrs_cd)
    #train loop 
    for epoch in tqdm(range(cfg.epochs)):
        for itr, data in enumerate(train_loader):
            
            gt_mask = data['mask'].to(device)
            
            input_img = data['img'].to(device)
            
            seg_out = model(input_img)

            #TODO: verify the dim of the softmax dim
            # seg_loss = criterion(F.log_softmax(seg_out, dim =1), gt_mask)

            #TODO: add a condition of lane exist loss

            
        

    #eval Loop 















