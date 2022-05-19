import dis
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

#build import for different moduless
from datasets.Apollo3d_loader import Apollo3d_loader, collate_fn
from models.build_model import load_model
from utils.helper_functions import *
from anchorless_detector import load_3d_model

def classification_regression_loss(rho_pred, rho_gt, delta_z_pred, delta_z_gt, cls_pred, cls_gt, phi_pred, phi_gt ):
        """"
        Params:
            rho_pred: predicted rho [batch_size,13,8]
            rho_gt: ground truth rho [batch_size,13,8]
            delta_z_pred: predicted delta_z [batch_size,13,8]
            delta_z_gt: ground truth delta_z [batch_size,13,8]
            cls_pred: predicted cls [batch_size,13,8]
            cls_gt: ground truth cls [batch_size,13,8]
            phi_pred: predicted phi [batch_size,10,13,8]
            phi_gt: ground truth phi [batch_size,10,13,8]

        return:
            Angle_loss: loss for angle regression (Cross entropy loss for phi vector) + l1 loss for phi vector
            offset_loss: l1 loss for delta_z + l1 loss for rho
            score_loss: BCE loss for cls_score regression

            Overall_loss: score_loss + c_ij * Angle_loss + c_ij * offset_loss
        """

        L1loss= nn.L1Loss()
        BCEloss = nn.BCEWithLogitsLoss()
        CEloss = nn.CrossEntropyLoss()
        
        batch_size = rho_pred.shape[0]

        #VALIDATE & TODO: manage the datatypes later for the loss calculation for training 
        Overall_loss = torch.tensor(0, dtype = cls_pred.dtype, device = cls_pred.device)
        
        #TODO: change the condition for loops(as per the shape of the tile grid)
        for b in range(rho_pred.shape[0]):
            for i in range(rho_pred.shape[1]): # 13 times 
                for j in range(rho_pred.shape[2]): # 8 times
                    #----------------- Offsets loss----------
                    loss_rho_ij = L1loss(rho_pred[b,i,j],rho_gt[b,i,j])
                    loss_delta_z_ij = L1loss(delta_z_pred[b,i,j], delta_z_gt[b,i,j])
                    
                    offsetsLoss_ij = loss_rho_ij + loss_delta_z_ij 
                    
                    #----------------classification score loss---------
                    loss_score_ij = BCEloss(cls_pred[b,i,j], cls_gt[b,i,j])
                    
                    #--------------- Line angle loss ------------------
                    #TODO: add delta phi loss with indicator function
                    loss_phi_ij = CEloss(phi_pred[b,:,i,j].reshape(1,10),phi_gt[b,:,i,j].reshape(1,10))
                    
                    Lineangle_loss = loss_phi_ij
                    
                    #----------------Overall loss -------------------
                    Overall_loss_ij =  loss_score_ij + cls_gt[b,i,j]* Lineangle_loss + cls_gt[b,i,j] * offsetsLoss_ij
                    
                    Overall_loss = Overall_loss + Overall_loss_ij 
            
    #         #TODO: Verify if I need to divid this loss for one batch by grid_w * grid_h
    #         Overall_loss = Overall_loss/ (rho_pred.shape[1]* rho_pred.shape[2])
        Average_OverallLoss = Overall_loss / batch_size
        
        return Average_OverallLoss

def discriminative_loss(embedding, delta_c_gt, cfg, device = None):  
    
    """
    Arguments:
    Embedding == f_ij 
    delta_c = del_i,j for classification if that tile is part of the lane or not
    tile_size = (grid_w, grid_h) --> squre patch
    
    return:
    clustering loss/ clustering loss (aka), push and pull loss
    """

    pull_loss = torch.tensor(0 ,dtype = embedding.dtype, device = embedding.device)
    push_loss = torch.tensor(0, dtype = embedding.dtype, device = embedding.device)
    
    #iterating over batches
    for b in range(embedding.shape[0]):
        
        embedding_b = embedding[b]  #---->(4,H*,W*)
        delta_c_gt_b = delta_c_gt[b] # will be a tensor of size (13,8) or whatever the grid size is consits of lane labels
        
        #delta_c_gt ---> [batch_size, 13, 8] where every element tells you which lane you belong too. 

        ##TODO: Add condition for 0 class
        labels = torch.unique(delta_c_gt_b) #---> array of type of labels
        num_lanes = len(labels)
        
        if num_lanes==0:
            _nonsense = embedding.sum()
            _zero = torch.zeros_like(_nonsense)
            pull_loss = pull_loss + _nonsense * _zero
            push_loss = push_loss + _nonsense * _zero
            continue

        centroid_mean = []
        for lane_c in labels: # it will run for the number of lanes basically l_c = 1,2,3,4,5 
            
            #1. Obtain one hot tensor for tile class labels
            delta_c = torch.where(delta_c_gt_b==lane_c,1,0) # bool tensor for lane_c ----> size (13,8)

            tensor, count = torch.unique(delta_c, return_counts=True)
            N_c = count[1].item() # number of tiles in lane_c
            
            patchwise_mean = []
            
            #extracting tile patches from the embedding tensor
            for r in range(0,embedding_b.shape[1],cfg.tile_size):
                for c in range(0,embedding_b.shape[2],cfg.tile_size):
                    
                    f_ij = embedding_b[:,r:r+cfg.tile_size,c:c+cfg.tile_size] #----> (4,32,32) 
                    f_ij = f_ij.reshape(f_ij.shape[0], f_ij.shape[1]*f_ij.shape[2])
                    
                    #2. calculate mean for lane_c (mu_c) patchwise
                    mu_c = torch.sum(f_ij * delta_c[int(r/cfg.tile_size),int(c/cfg.tile_size)], dim = 1)/N_c #--> (4) mu for all the four embeddings
                    patchwise_mean.append(mu_c)
                    #3. calculate the pull loss patchwise
                    
                    pull_loss = pull_loss + torch.mean(F.relu( delta_c[int(r/cfg.tile_size),int(c/cfg.tile_size)] * torch.norm(f_ij-mu_c.reshape(4,1),dim = 0)- cfg.delta_pull)**2) / num_lanes
                    
            patchwise_centroid = torch.stack(patchwise_mean) #--> (32*32,4)
            patchwise_centroid = torch.mean(patchwise_centroid, dim =0) #--> (4)
            
            centroid_mean.append(patchwise_centroid)

        centroid_mean = torch.stack(centroid_mean) #--> (num_lanes,4)

        if num_lanes > 1:
            
            #4. calculate the push loss
            centroid_mean_A = centroid_mean.reshape(-1,1, 4)
            centroid_mean_B =centroid_mean.reshape(1,-1, 4)

            dist = torch.norm(centroid_mean_A-centroid_mean_B, dim = 2) #--> (num_lanes,num_lanes)
            dist = dist + torch.eye(num_lanes, dtype = dist.dtype, device = dist.device) * cfg.delta_push
            
            #divide by 2 to compensate the double loss calculation
            push_loss = push_loss + torch.sum(F.relu(-dist + cfg.delta_push)**2) / (num_lanes * (num_lanes-1)) / 2
    
    pull_loss= pull_loss / cfg.batch_size
    push_loss = push_loss / cfg.batch_size

    return pull_loss, push_loss

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
    parser.add_argument("--dataset_type", type = str, default = "Apollo3d", help = "Dataset type")
    parser.add_argument("--config", type=str, default="configs/config_anchorless_3dlane.py", help="config file")
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--seed", type=int, default=27, help="random seed")
    parser.add_argument("--baseline", type=bool, default=False, help="enable baseline")
    parser.add_argument("--pretrained2d", type=bool, default=False, help="enable pretrained 2d lane detection model")
    parser.add_argument("--pretrained3d", type=bool, default=False, help="enable pretrained anchorless 3d lane detection model")
    parser.add_argument("--data_dir", type=str, default="/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/Apollo_Sim_3D_Lane_Release", help="data directory")
    parser.add_argument("--data_split", type=str, default="standard", help="data split")
    parser.add_argument("--path_data_split", type=str, default="/home/gautam/e2e/lane_detection/3d_approaches/3d_dataset/3D_Lane_Synthetic_Dataset/old_data_splits", help="path to data split")
    parser.add_argument("--e2e",type=bool, default=False, help="enable end-to-end training")
    
    #parsing args
    args = parser.parse_args()

    #load config file
    cfg = Config.fromfile(args.config)

    # for reproducibility
    torch.manual_seed(args.seed)

    #trained model paths
    checkpoints_dir = './nets/3dlane_detection' + '/' + args.dataset_type + '/checkpoints'
    result_model_dir = './nets/3dlane_detection' + '/' + args.dataset_type + '/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    if args.dataset_type == "Apollo3d":
        if args.data_split == "standard":
            data_split = os.path.join(args.path_data_split, "standard")
        elif args.data_split == "rare_subset":
            data_split = os.path.join(args.path_data_split, "rare_subset")
        elif args.data_split == "illus_chng":
            data_split = os.path.join(args.path_data_split, "illus_chng")

        data_root = args.data_dir
    else:
        #TODO: add the arguments Later for OpenLane dataset
        pass

    #dataloader
    train_dataset = Apollo3d_loader(data_root, data_split, cfg = cfg, phase = "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn = collate_fn, pin_memory=True)

    # #this will be used for validation and evaluation at the same time
    # test_dataset = Apollo3d_loader(args.data_dir, args.data_split, phase = "test")
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    #load model and weights
    if args.e2e == True:
        #load 2d model from checkpoint and train the whole pipeline end-to-end
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device) 
    else: 
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device)
    
    """
    Model related TODO's for End-to-End training
    #TODO: Add all the parameters to the config file
    #TODO: Have some sort of logic if 2d model will be trained or not with the other network
    """
    
    #NOTE:: Currently both the schedulers have same parameters:: Separate them IF needed
    #NOTE: if args.pretrained2d == "False" and args.pretrained3d == "False" the network will be trained end to end
     
    #optimizer and scheduler
    if args.pretrained2d == "False":
        param_group1 = model2d.parameters()
        optimizer1 = topt.Adam(param_group1, cfg.lr, weight_decay=cfg.l2_lambda)
        scheduler1 = topt.lr_scheduler.ReduceLROnPlateau(optimizer1, factor = cfg.lrs_factor, patience = cfg.lrs_patience, threshold=cfg.lrs_thresh,
                                                        verbose=True, min_lr=cfg.lrs_min, cooldown=cfg.lrs_cd)
    
    elif args.pretrained3d == "False":
        param_group2 = model3d.parameters()
        optimizer2 = topt.Adam(param_group2, cfg.lr, weight_decay=cfg.l2_lambda)
        scheduelr2 = topt.lr_scheduler.ReduceLROnPlateau(optimizer2, factor = cfg.lrs_factor, patience = cfg.lrs_patience, threshold=cfg.lrs_thresh,
                                                        verbose=True, min_lr=cfg.lrs_min, cooldown=cfg.lrs_cd)
    
    # #training loop
    # #unit test
    # a = torch.rand(1,3,360,480).to(device)

    # o = model2d(a)
    # print("checking the shape of the output tensor from 2d lane seg pipeline",o.shape)

    # o = o.softmax(dim=1)
    # o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
    # o = o[:,1:2,:,:]    
    # """
    # #BIG BIG NOTE: In my case remember I need to obtain the binary seg masks for the other network output 
    # currently I have 7 classes Investigate that when the network is finished
    # """

    # o2 = model3d(o)
    # print("checking the shape of the output tensor",o2.shape)
    
    """
    TODO: Make the functions for training and validation
    """

    #train_loop
    for itr, data in enumerate(train_loader):
        
        #get the data
        batch = {}
        batch.update({"input_image":data[0].to(device),
                      "aug_mat":data[1],
                      "gt_height":data[2],
                      "gt_pitch":data[3],
                      "gt_lane_points":data[4],
                      "gt_rho":data[5].to(device),
                      "gt_phi":data[6].to(device),
                      "gt_cls_score":data[7].to(device),
                      "gt_lane_cls":data[8].to(device),
                      "gt_delta_z":data[9].to(device)})

        #forward pass
        #TODO: change the last layer of the 2d model
       
        o = model2d(batch["input_image"])
        o = o.softmax(dim=1)
        o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
        print("shape of o before max", o.shape)
        o = o[:,1:2,:,:]

        print("checking the shape of o", o.shape)

        out1 = model3d(o)

        out_pathway1 = out1["embed_out"]
        out_pathway2 = out1["bev_out"]

        rho_pred = out_pathway2[:,0,...]
        delta_z_pred = out_pathway2[:,1,...]
        cls_score_pred = out_pathway2[:,2,...]
        phi_pred = out_pathway2[:,3:,...]
        
        print(phi_pred.dtype)
        print(batch["gt_phi"].dtype)

        phi_gt = torch.tensor(batch["gt_phi"], dtype = torch.long, device = device)


        # print("rho_pred shape",rho_pred.shape)
        # print("delta_z_pred shape",delta_z_pred.shape)
        # print("cls_score_pred shape",cls_score_pred.shape)
        print("phi_pred shape",phi_pred[:,:,0,0])

        # print("gt_rho shape",batch["gt_rho"].shape)
        # print("gt_delta_z shape",batch["gt_delta_z"].shape)
        # print("gt_cls_score shape",batch["gt_cls_score"].shape)
        print("gt_phi shape",batch["gt_phi"][:,:,0,0])


        loss1 = discriminative_loss(out1["embed_out"], batch["gt_lane_cls"],cfg)
        print(loss1)
        loss2 = classification_regression_loss(rho_pred, batch["gt_rho"], delta_z_pred, batch["gt_delta_z"], cls_score_pred, batch["gt_cls_score"], phi_pred, phi_gt)
        print(loss2)

