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


#build import for different moduless
from datasets.Apollo3d_loader import Apollo3d_loader, collate_fn, Visualization
from models.build_model import load_model
from utils.helper_functions import *
from anchorless_detector import load_3d_model

def classification_regression_loss(L1loss, BCEloss, CEloss, rho_pred, rho_gt, delta_z_pred, delta_z_gt, cls_pred, cls_gt, phi_pred, phi_gt ):
        """"
        Params: (Here 13x8 is the grid size of tiles as per the spatial size of one tile)
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
            score_loss: BCEwithLogits loss for cls_score regression

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

    #         #TODO: Verify if I need to divide this loss for one batch by grid_w * grid_h
            # Overall_loss = Overall_loss/ (rho_pred.shape[1]* rho_pred.shape[2])
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
    
    pull_loss = pull_loss / cfg.batch_size
    push_loss = push_loss / cfg.batch_size

    loss_embedding = pull_loss + push_loss 
    return loss_embedding # batch loss


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
    parser.add_argument("--pretrained2d", type=bool, default=True, help="enable pretrained 2d lane detection model")
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

    # # # TODO: split the test set into val and test set after the training is verified.
    val_dataset = Apollo3d_loader(data_root, data_split, cfg = cfg, phase = "test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn = collate_fn, pin_memory=True)
    
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    print("===> batches in train loader", train_loader_len)
    print("===> batches in val loader", val_loader_len)
    
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
    #NOTE: verify that BCEWithLogitsLoss for score as We know that when the last layer has acitvations normal loss can be used (numerically stable rn) 
    BCEloss = nn.BCEWithLogitsLoss().to(device)
    CEloss = nn.CrossEntropyLoss().to(device)
    m = nn.Sigmoid()

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
        scheduler2 = topt.lr_scheduler.ReduceLROnPlateau(optimizer2, factor = cfg.lrs_factor, patience = cfg.lrs_patience, threshold=cfg.lrs_thresh,
                                                           verbose=True, min_lr=cfg.lrs_min, cooldown=cfg.lrs_cd)
    else: 
        print("===> Using pretrained 3d model") 
    
    #### NOTE: TODO: ADD conditions everywhere for e2e and normal training

    #train_loop
    print("======> Starting to train")
    
    #TODO: add autograd profiler for the training: false to speed up the training
    for epoch in tqdm(range(cfg.epochs)):

        batch_loss = 0.0 
        tr_loss = 0.0 
        start_point = time.time()

        timings = dict()
        multitimings = MultiTiming(timings)

        for itr, data in enumerate(train_loader):
            
            if args.e2e == True:
                model2d.train()
                model3d.train()
            else: 
                model2d.eval()
                model3d.train()

            #flag for train log and validation loop
            ## TODO: change it to per epoch not iteration
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
            # print(cls_score_pred.round())
            phi_pred = out_pathway2[:,3:,...] 

            loss1 = discriminative_loss(out1["embed_out"], batch["gt_lane_cls"],cfg)
            loss2 = classification_regression_loss(L1loss, BCEloss, CEloss, rho_pred, batch["gt_rho"], delta_z_pred, batch["gt_delta_z"], cls_score_pred, batch["gt_cls_score"], phi_pred, batch["gt_phi"])
            
            # print("==>discriminative loss::", loss1.item())
            # print("==>classification loss::", loss2.item())
            overall_loss = cfg.w_clustering_Loss * loss1 + cfg.w_classification_Loss * loss2
            print("==>overall loss::", overall_loss.item())
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
            
            # eval loop
            if should_run_valid:

                model2d.eval()
                #TODO: check this issue of missing outputs in .eval()
                # model3d.eval()
                model3d.train()
                print(">>>>>>>Validating<<<<<<<<")

                val_loss = 0.0
                val_batch_loss = 0.0

                with torch.no_grad():
                    for val_itr, val_data in enumerate(val_loader):
                        val_batch = {}
                        val_batch.update({"input_image":val_data[0].to(device),
                                    "gt_height":val_data[1].to(device),
                                    "gt_pitch":val_data[2].to(device),
                                    "gt_lane_points":val_data[3],
                                    "gt_rho":val_data[4].to(device),
                                    "gt_phi":val_data[5].to(device).float(),
                                    "gt_cls_score":val_data[6].to(device),
                                    "gt_lane_cls":val_data[7].to(device),
                                    "gt_delta_z":val_data[8].to(device)})

                        #update projection
                        model3d.update_projection(cfg, val_batch["gt_height"], val_batch["gt_pitch"])
                        
                        val_o = model2d(batch["input_image"])
                        val_o = val_o.softmax(dim=1)
                        val_o = val_o/torch.max(torch.max(val_o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
                        # print("shape of o before max", o.shape)
                        val_o = val_o[:,1:,:,:]

                        val_out1 = model3d(val_o)

                        val_out_pathway1 = val_out1["embed_out"]
                        val_out_pathway2 = val_out1["bev_out"]

                        val_rho_pred = val_out_pathway2[:,0,...]
                        val_delta_z_pred = val_out_pathway2[:,1,...]
                        val_cls_score_pred = val_out_pathway2[:,2,...]
                        print(val_cls_score_pred)
                        val_phi_pred = val_out_pathway2[:,3:,...]

                        val_loss1 = discriminative_loss(val_out1["embed_out"], val_batch["gt_lane_cls"],cfg)
                        val_loss2 = classification_regression_loss(L1loss, BCEloss, CEloss, val_rho_pred, val_batch["gt_rho"], val_delta_z_pred, val_batch["gt_delta_z"], val_cls_score_pred, val_batch["gt_cls_score"], val_phi_pred, val_batch["gt_phi"])
                    
                        val_overall_loss = cfg.w_clustering_Loss * val_loss1 + cfg.w_classification_Loss * val_loss2
                        
                        val_batch_loss = val_overall_loss.detach().cpu() / cfg.batch_size
                        val_loss += val_batch_loss
                    
                        if (val_itr +1) % 1 == 0:
                            val_running_loss = val_loss.item() / (val_itr + 1)
                            print(f"Validation: {val_itr+1} steps of ~{val_loader_len}.  Validation Running Loss {val_running_loss:.4f}")
                        

                        
                        """
                        TODO: 
                        1. Open the whole pred json file or an array where the points are stored
                        2. Calacualate metric here and return the metric value and create a model checkpoint here as per the metric value
                        """

                    val_avg_loss = val_loss / (val_itr +1)
                    print(f"Validation Loss: {val_avg_loss}")
                    
                    #TODO: add a condition for e2e if train whole model
                    scheduler2.step(val_avg_loss.item())

            if should_run_vis:
                
                print(">>>>>>>Visualizing<<<<<<<<")
                vis = Visualization(cfg.org_h, cfg.org_w, cfg.resize_h, cfg.resize_w, cfg.K, cfg.ipm_w, cfg.ipm_h, cfg.crop_y, cfg.top_view_region)
                
                with torch.no_grad():
                    
                    model2d.train()
                    model3d.eval()

                    for vis_itr, vis_data in enumerate(val_loader):
                        vis_batch = {}
                        vis_batch.update({"vis_gt_height":vis_data[1].cpu().numpy(),
                                        "vis_gt_pitch":vis_data[2].cpu().numpy(),
                                        "gt_lane_points":vis_data[3],
                                        "image_full_path":vis_data[9],
                                        "input_image":vis_data[0].to(device),
                                        "gt_height":vis_data[1].to(device),
                                        "gt_pitch":vis_data[2].to(device)})

                        #update projection
                        model3d.update_projection(cfg, vis_batch["gt_height"], vis_batch["gt_pitch"])
                        
                        vis_o = model2d(vis_batch["input_image"])
                        vis_o = vis_o.softmax(dim=1)
                        vis_o = vis_o/torch.max(torch.max(vis_o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
                        # print("shape of o before max", o.shape)
                        vis_o = vis_o[:,1:,:,:]

                        vis_out1 = model3d(vis_o)

                        vis_out_pathway2 = vis_out1["bev_out"]

                        vis_rho_pred = vis_out_pathway2[:,0,...] #---> (b,13,8)
                        vis_delta_z_pred = vis_out_pathway2[:,1,...] #--> (b,13,8)
                        vis_cls_score_pred = vis_out_pathway2[:,2,...] # --> (b,13,8)
                        # print(vis_cls_score_pred)
                        vis_phi_pred = vis_out_pathway2[:,3:,...] # --> (b,10,13,8) ---> (b,13,8)
                        
                        #TODO: make a separate function for this part of vis later-
                        gt_vis_ipm_images= [] 
                        pred_vis_ipm_images = []

                        gt_vis_2d_images = []
                        pred_vis_2d_images = []
                
                        for b in range(cfg.batch_size):
                            vis_img_path = vis_batch["image_full_path"][b]
                            vis_img = cv2.imread(vis_img_path)
                            
                            
                            """
                            HERE TODO: generate the 3D lane points from the prediction of one sample and add it to the visualizaton 
                            """
                            vis_rho_pred_b = vis_rho_pred[b,:,:].detach().cpu().numpy()
                            vis_phi_pred_b = vis_phi_pred[b,:,:,:].detach().cpu().numpy()
                            vis_delta_z_pred_b = vis_delta_z_pred[b,:,:].detach().cpu().numpy()
                            vis_cls_score_pred_b = vis_cls_score_pred[b,:,:].detach().cpu().numpy()
                            
                            # print(vis_cls_score_pred_b)
                            # vis_cls_score_pred_b = vis_cls_score_pred_b.round() # probs to 0 or 1

                            points = np.empty((0,3)) # N x 3 array of points

                            #unormalize the rho and delta z
                            vis_rho_pred_b = vis_rho_pred_b * (cfg.max_lateral_offset - cfg.min_lateral_offset) + cfg.min_lateral_offset
                            vis_delta_z_pred_b = vis_delta_z_pred_b * (cfg.max_delta_z - cfg.min_delta_z) + cfg.min_delta_z

                            vis_cam_height_b = vis_batch["vis_gt_height"][b]
                            vis_cam_pitch_b = vis_batch["vis_gt_pitch"][b]

                            for i in range(vis_rho_pred_b.shape[0]):
                                for j in range(vis_rho_pred_b.shape[1]):
                                    
                                    ##TODO: Turn this if onn later for the final submission
                                    # if vis_cls_score_pred_b[i,j] == 1:
                                    #extract points from predictions
                                    rho = vis_rho_pred_b[i,j]
                                    phi_vec = vis_phi_pred_b[:,i,j] # ---> 1d array of 10 elements containing probs
                                    phi = palpha2alpha(phi_vec)
                                    delta_z = vis_delta_z_pred_b[i,j]

                                    lane_point = polar_to_catesian(phi, vis_cam_pitch_b, vis_cam_height_b, delta_z, rho)
                                    # print("rho", rho)
                                    # print("phi", phi)
                                    # print("delta_z",delta_z)

                                    # print("lane point", lane_point)
                                    points = np.append(points , lane_point.reshape(1,3), axis = 0)

                            #list containing arrays of lane points
                            gt_ipm, gt_2d = vis.draw_lanes(vis_batch["gt_lane_points"][b], vis_img, vis_batch["vis_gt_height"][b], vis_batch["vis_gt_pitch"][b])         
                            
                            #obtain the similar thing for the predicted lane points

                            # cv2.imwrite("ipm_test.jpg",gt_ipm)
                            # cv2.imwrite("2d_test.jpg",gt_2d)
                            
                            gt_vis_ipm_images.append(gt_ipm)
                            gt_vis_2d_images.append(gt_2d)

                            break #visualize only one sample for now per vis iteration
                        break













