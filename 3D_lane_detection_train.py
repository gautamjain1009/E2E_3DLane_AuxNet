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

def classification_regression_loss(L1loss, BCEloss, CEloss, rho_pred, rho_gt, delta_z_pred, delta_z_gt, cls_pred, cls_gt, phi_pred, phi_gt ):
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
        m = nn.Softmax(dim =1)
        
        batch_size = rho_pred.shape[0]
        Overall_loss = torch.tensor(0, dtype = cls_pred.dtype, device = cls_pred.device)
        
        for b in range(rho_pred.shape[0]):
            for i in range(rho_pred.shape[1]): # 13 times 
                for j in range(rho_pred.shape[2]): # 8 times
                    #----------------- Offsets loss----------
                    loss_rho_ij = L1loss(rho_pred[b,i,j],rho_gt[b,i,j])
                    loss_delta_z_ij = L1loss(delta_z_pred[b,i,j], delta_z_gt[b,i,j])
                    
                    offsetsLoss_ij = loss_rho_ij + loss_delta_z_ij 
                    
                    #----------------classification score loss---------
                    loss_score_ij = BCEloss(cls_pred[b,i,j], cls_gt[b,i,j])
                                        
                    # --------------- Line angle loss ------------------
                    #TODO: add delta phi loss with indicator function
                    phi_gt_ij = m(phi_gt[b,:,i,j].reshape(1,10))
                    loss_phi_ij = CEloss(phi_pred[b,:,i,j].reshape(1,10),phi_gt_ij)
                    
                    # print("loss_phi_ij", loss_phi_ij)
                    Lineangle_loss = loss_phi_ij

                    #----------------Overall loss -------------------
                    # Overall_loss_ij =  loss_score_ij + cls_gt[b,i,j] * offsetsLoss_ij
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

    loss_embedding = pull_loss + push_loss
    return loss_embedding


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

    # # TODO: split the test set into val and test set after the training is verified.
    # val_dataset = Apollo3d_loader(args.data_dir, args.data_split, phase = "test")
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    train_loader_len = len(train_loader)
    # val_loader_len = len(val_loader)

    print("===> batches in train loader", train_loader_len)
    # print("===> batches in val loader", val_loader_len)
    
    #load model and weights
    if args.e2e == True: #TODO: make a single forward function for the combined model
        #load 2d model from checkpoint and train the whole pipeline end-to-end
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device) 
    else: 
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device)
    
    #general loss functions
    L1loss= nn.L1Loss().to(device)
    #NOTE: verify that BCEWithLogitsLoss for score as We know that when the last layer has aa citvations normal loss can be used (numerically stable rn) 
    BCEloss = nn.BCEWithLogitsLoss().to(device)
    CEloss = nn.CrossEntropyLoss().to(device)
    """
    Model related TODO's for End-to-End training
    #TODO: Add all the parameters to the config file
    #TODO: Have some sort of logic if 2d model will be trained or not with the other network
    """
    
    #NOTE:: Currently both the schedulers have same parameters:: Separate them IF needed
    #NOTE: if args.pretrained2d == "False" and args.pretrained3d == "False" the network will be trained end to end
     
    #optimizer and scheduler
    if not args.pretrained2d:
        print("====> initialized optimzer and scheduler for 2d model")
        param_group1 = model2d.parameters()
        optimizer1 = topt.Adam(param_group1, cfg.lr, weight_decay=cfg.l2_lambda)
        scheduler1 = topt.lr_scheduler.ReduceLROnPlateau(optimizer1, factor = cfg.lrs_factor, patience = cfg.lrs_patience, threshold=cfg.lrs_thresh,
                                                        verbose=True, min_lr=cfg.lrs_min, cooldown=cfg.lrs_cd)
    else: 
        print("===> Using pretrained binary segmentaiton model")
    
    if not args.pretrained3d:
        print("====> initialized optimzer and schdeuler for 3d model")
        param_group2 = model3d.parameters()
        optimizer2 = topt.Adam(param_group2, cfg.lr, weight_decay=cfg.l2_lambda)
        scheduelr2 = topt.lr_scheduler.ReduceLROnPlateau(optimizer2, factor = cfg.lrs_factor, patience = cfg.lrs_patience, threshold=cfg.lrs_thresh,
                                                           verbose=True, min_lr=cfg.lrs_min, cooldown=cfg.lrs_cd)
    else: 
        print("===> Using pretrained 3d model")

    #TODO: will be added in the train function later
 
    
    #### NOTE: TODO: ADD conditions everywhere for e2e and normal training

    #train_loop
    print("======> Starting to train")
    
    #TODO: add autograd profiler for the training: false to speed up the training
    for epoch in tqdm(range(cfg.epochs)):

        batch_loss = 0.0 
        tr_loss = 0.0 
        start_point = time.time()

        # timings = dict()
        # multitimings = MultiTiming(timings)


        for itr, data in enumerate(train_loader):
            
            if args.e2e == True:
                model2d.train()
                model3d.train()
            else: 
                model2d.eval()
                model3d.train()

            #flag for train log and validation loop
            should_log_train = (itr+1) % cfg.train_log_frequency == 0 
            should_run_valid = (itr+1) % cfg.val_frequency == 0
            should_run_vis = (itr+1) % cfg.val_frequency == 0
            
            #get the data
            batch = {}
            batch.update({"input_image":data[0].to(device),
                        "aug_mat":data[1].to(device).float(),
                        "gt_height":data[2].to(device),
                        "gt_pitch":data[3].to(device),
                        "gt_lane_points":data[4],
                        "gt_rho":data[5].to(device),
                        "gt_phi":data[6].to(device).float(),
                        "gt_cls_score":data[7].to(device),
                        "gt_lane_cls":data[8].to(device),
                        "gt_delta_z":data[9].to(device)})

            """
            update projection
            update augmented matrix
            optimzer.zero_grad() condition with e2e and simple  
            """
            print(batch["gt_height"].dtype)
            #TODO: add the condition for camera fix
            #update projection
            model3d.update_projection(cfg, batch["gt_height"], batch["gt_pitch"])

            #update augmented matrix
            model3d.update_projection_for_data_aug(batch["aug_mat"])

            optimizer2.zero_grad(set_to_none= True)

            #forward pass
            o = model2d(batch["input_image"])
            o = o.softmax(dim=1)
            o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
            # print("shape of o before max", o.shape)
            o = o[:,1:,:,:]

            out1 = model3d(o)

            out_pathway1 = out1["embed_out"]
            out_pathway2 = out1["bev_out"]

            rho_pred = out_pathway2[:,0,...]
            delta_z_pred = out_pathway2[:,1,...]
            cls_score_pred = out_pathway2[:,2,...]
            phi_pred = out_pathway2[:,3:,...]
        
            loss1 = discriminative_loss(out1["embed_out"], batch["gt_lane_cls"],cfg)
            loss2 = classification_regression_loss(L1loss, BCEloss, CEloss, rho_pred, batch["gt_rho"], delta_z_pred, batch["gt_delta_z"], cls_score_pred, batch["gt_cls_score"], phi_pred, batch["gt_phi"])
        
            w_clustering_Loss = 0.3
            w_classification_Loss = 0.7

            overall_loss = w_clustering_Loss * loss1 + w_classification_Loss * loss2

            overall_loss.backward()
            optimizer2.step()

            batch_loss = overall_loss.detach().cpu() / cfg.batch_size

            #reporting model fps
            # fps = cfg.batch_size / (time.time() - start_po

            tr_loss += batch_loss

            if should_log_train:

                running_loss = tr_loss.item() / cfg.train_log_frequency
                print(f"Epoch: {epoch+1}/{cfg.epochs}. Done {itr+1} steps of ~{train_loader_len}. Running Loss:{running_loss:.4f}")
                # pprint(timings)

                #TODO: log results on wandb


                tr_loss  = 0.0
            
            #eval loop
            if should_run_valid:
                """
                val loop with the loader and make the model checkpoint
                
                """
                    model2d.eval()
                    model3d.eval()
                    print(">>>>>>>Validating<<<<<<<<")

                    val_loss = 0.0
                    val_batch_loss = 0.0

                    with torch.no_grad():
                        for val_itr, val_data in enumerate(val)
                

            # if should_run_vis:
            #     """
            #     vis loop with the loader with 2d, BEV visualisation and 3d visualisation for gt and predictions
            #     """
        




