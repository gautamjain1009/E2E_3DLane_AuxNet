import warnings

warnings.filterwarnings("ignore", category=UserWarning, message='Length of IterableDataset')
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
from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt
from timing import * 
import json
from moviepy.video.io.bindings import mplfig_to_npimage
#build import for different moduless
from datasets.Apollo3d_loader import Apollo3d_loader, Visualization, configure_worker, BatchDataLoader, BackgroundGenerator
from models.build_model import load_model
from utils.helper_functions import *
from anchorless_detector import load_3d_model
from evaluate import Apollo_3d_eval
import gc
from torchvision import transforms
import matplotlib.pyplot as plt

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

# NOTE: Discriminative loss for embedding feaetures without max pool
# def discriminative_loss(embedding, delta_c_gt, cfg, device = None):  
    
#     """
#     Arguments:
#     Embedding == f_ij 
#     delta_c = del_i,j for classification if that tile is part of the lane or not
#     tile_size = (grid_w, grid_h) --> squre patch
    
#     return:
#     clustering loss/ clustering loss (aka), push and pull loss
#     """

#     pull_loss = torch.tensor(0 ,dtype = embedding.dtype, device = embedding.device)
#     push_loss = torch.tensor(0, dtype = embedding.dtype, device = embedding.device)
    
#     #iterating over batches
#     for b in range(embedding.shape[0]):
        
#         embedding_b = embedding[b]  #---->(4,H*,W*)
#         delta_c_gt_b = delta_c_gt[b] # will be a tensor of size (13,8) or whatever the grid size is consits of lane labels
        
#         #delta_c_gt ---> [batch_size, 13, 8] where every element tells you which lane you belong too. 

#         ##TODO: Add condition for 0 class
#         labels = torch.unique(delta_c_gt_b) #---> array of type of labels
#         num_lanes = len(labels)
        
#         if num_lanes==0:
#             _nonsense = embedding.sum()
#             _zero = torch.zeros_like(_nonsense)
#             pull_loss = pull_loss + _nonsense * _zero
#             push_loss = push_loss + _nonsense * _zero
#             continue

#         centroid_mean = []
#         for lane_c in labels: # it will run for the number of lanes basically l_c = 1,2,3,4,5 
            
#             #1. Obtain one hot tensor for tile class labels
#             delta_c = torch.where(delta_c_gt_b==lane_c,1,0) # bool tensor for lane_c ----> size (13,8)

#             tensor, count = torch.unique(delta_c, return_counts=True)
#             N_c = count[1].item() # number of tiles in lane_c
            
#             patchwise_mean = []
            
#             #extracting tile patches from the embedding tensor
#             for r in range(0,embedding_b.shape[1],cfg.tile_size):
#                 for c in range(0,embedding_b.shape[2],cfg.tile_size):
                    
#                     f_ij = embedding_b[:,r:r+cfg.tile_size,c:c+cfg.tile_size] #----> (4,32,32) 
#                     f_ij = f_ij.reshape(f_ij.shape[0], f_ij.shape[1]*f_ij.shape[2])
                    
#                     #2. calculate mean for lane_c (mu_c) patchwise
#                     mu_c = torch.sum(f_ij * delta_c[int(r/cfg.tile_size),int(c/cfg.tile_size)], dim = 1)/N_c #--> (4) mu for all the four embeddings
#                     patchwise_mean.append(mu_c)
#                     #3. calculate the pull loss patchwise
                    
#                     pull_loss = pull_loss + torch.mean(F.relu( delta_c[int(r/cfg.tile_size),int(c/cfg.tile_size)] * torch.norm(f_ij-mu_c.reshape(4,1),dim = 0)- cfg.delta_pull)**2) / num_lanes
                    
#             patchwise_centroid = torch.stack(patchwise_mean) #--> (32*32,4)
#             patchwise_centroid = torch.mean(patchwise_centroid, dim =0) #--> (4)
            
#             centroid_mean.append(patchwise_centroid)

#         centroid_mean = torch.stack(centroid_mean) #--> (num_lanes,4)

#         if num_lanes > 1:
            
#             #4. calculate the push loss
#             centroid_mean_A = centroid_mean.reshape(-1,1, 4)
#             centroid_mean_B =centroid_mean.reshape(1,-1, 4)

#             dist = torch.norm(centroid_mean_A-centroid_mean_B, dim = 2) #--> (num_lanes,num_lanes)
#             dist = dist + torch.eye(num_lanes, dtype = dist.dtype, device = dist.device) * cfg.delta_push
            
#             #divide by 2 to compensate the double loss calculation
#             push_loss = push_loss + torch.sum(F.relu(-dist + cfg.delta_push)**2) / (num_lanes * (num_lanes-1)) / 2
    
#     pull_loss = pull_loss / cfg.batch_size
#     push_loss = push_loss / cfg.batch_size

#     loss_embedding = pull_loss + push_loss 
#     return loss_embedding # batch loss

#NOTE: Discriminative loss for embedding feaetures with max pool
def discriminative_loss(embedding, seg_gt, cfg, device = None):
    """
    Arguments: 
    (H* = tile_height, W* = tile_width)
    Embedding == (1,4,H*,W*)
    seg_gt = lane class or background
    
    return:
    clustering loss/ clustering loss (aka), push, pull and regularize loss
    """
    batch_size = embedding.shape[0]

    pull_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device) #(var)
    push_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device) #(push)
    reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    
    for b in range(embedding.shape[0]):
        embedding_b = embedding[b] # (embed_dim, H, W)
        seg_gt_b = seg_gt[b]

        labels = torch.unique(seg_gt_b) # ---> array of type of labels
        labels = labels[labels!=0]
        num_lanes = len(labels)

        if num_lanes==0:
            _nonsense = embedding.sum()
            _zero = torch.zeros_like(_nonsense)
            pull_loss = pull_loss + _nonsense * _zero
            push_loss = push_loss + _nonsense * _zero
            reg_loss = reg_loss + _nonsense * _zero
            continue

        centroid_mean = []
        for lane_idx in labels: # it will run for the number of lanes basically l_c = 1,2,3,4,5 
            
            seg_mask_i = (seg_gt_b == lane_idx)
   
            if not seg_mask_i.any():
                continue
            embedding_i = embedding_b[:, seg_mask_i]

            mean_i = torch.mean(embedding_i, dim=1)
            centroid_mean.append(mean_i)

            # ---------- pull_loss -------------
            pull_loss = pull_loss + torch.mean( F.relu(torch.norm(embedding_i-mean_i.reshape(cfg.embedding_dim,1), dim=0) - cfg.delta_pull)**2 ) / num_lanes
        centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

        if num_lanes > 1:
            centroid_mean1 = centroid_mean.reshape(-1, 1, cfg.embedding_dim)
            centroid_mean2 = centroid_mean.reshape(1, -1, cfg.embedding_dim)

            dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
            dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * cfg.delta_push # diagonal elements are 0, now mask above delta_d

            # divided by two for double calculated loss above, for implementation convenience
            push_loss = push_loss + torch.sum(F.relu(-dist + cfg.delta_push)**2) / (num_lanes * (num_lanes-1)) / 2

        reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1)) #not used in the semi-local 3d lanenet

    pull_loss = pull_loss / batch_size
    push_loss = push_loss / batch_size
    reg_loss = reg_loss / batch_size    
    loss_embedding = pull_loss + push_loss + reg_loss

    return loss_embedding

def visualization(cfg, model2d, model3d, vis_loader, p, device, epoch, itr):

    print(">>>>>>>Visualizing<<<<<<<<")
    vis = Visualization(cfg.org_h, cfg.org_w, cfg.resize_h, cfg.resize_w, cfg.K, cfg.ipm_w, cfg.ipm_h, cfg.crop_y, cfg.top_view_region)
    
    model3d.eval()

    if cfg.visualize_activations: 
        model_weights = []
        conv_layers = []
        model_children = list(model3d.children())
        # print(model_children)
        counter =0 
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter+=1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for child in model_children[i].children():
                    # print(type(child))
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
        # print(conv_layers)

    with torch.no_grad():
        for vis_itr, vis_data in enumerate(vis_loader):
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
            
            vis_o = model2d(vis_batch["input_image"].contiguous().float())
            
            # a = torch.argmax(vis_o, dim =1)
            # print(torch.unique(a))

            vis_o = vis_o.softmax(dim=1)
            vis_o = vis_o/torch.max(torch.max(vis_o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
            # print("shape of o before max", o.shape)
            vis_o = vis_o[:,1:,:,:]

            vis_out = model3d(vis_o)
            
            vis_out_pathway1 = vis_out["embed_out"]
            vis_out_pathway2 = vis_out["bev_out"] #---(N, 4, H, W)
            vis_out_project = vis_out["project_out"]
            
            if cfg.visualize_activations:
                conv_layer_reg = conv_layers[:4]
                conv_layer_embed = conv_layers[4:]
                
                #activations of regression layer
                results_reg = [conv_layer_reg[0](vis_out_project[0:1,:,:,:])]  #only first sample of the batch is used for activation vis
                for i in range(1, len(conv_layer_reg)):
                    results_reg.append(conv_layer_reg[i](results_reg[-1]))
                outputs_reg = results_reg                    
                
                #activations of embedding layer
                results_embed = [conv_layer_embed[0](vis_out_project[0:1,:,:,:])]
                for i in range(1, len(conv_layer_embed)):
                    results_embed.append(conv_layer_embed[i](results_embed[-1]))
                outputs_embed = results_embed

                # for feature_map in outputs_embed:
                #     print(feature_map.shape)
                reg_processed = []
                for feature_map in outputs_reg:
                    feature_map = feature_map.squeeze(0)
                    gray_scale = torch.sum(feature_map,0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    reg_processed.append(gray_scale.data.cpu().numpy())
                    
                embed_processed = []
                for feature_map in outputs_embed:
                    feature_map = feature_map.squeeze(0)
                    gray_scale = torch.sum(feature_map,0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    embed_processed.append(gray_scale.data.cpu().numpy())
            
                reg_activation_fig = plt.figure(figsize=(30, 50))
                for i in range(len(reg_processed)):
                    a = reg_activation_fig.add_subplot(2, 2, i+1)
                    imgplot = plt.imshow(reg_processed[i])
                    a.axis("off")
                    a.set_title(str(i), fontsize=30)
                
                embed_activation_fig = plt.figure(figsize=(30, 50))
                for i in range(len(embed_processed)):
                    a = embed_activation_fig.add_subplot(1,2 , i+1)
                    imgplot = plt.imshow(embed_processed[i])
                    a.axis("off")
                    a.set_title(str(i), fontsize=30)
                
                reg_activation_fig = mplfig_to_npimage(reg_activation_fig)
                embed_activation_fig = mplfig_to_npimage(embed_activation_fig)

                wandb.log({"Embedding Activations_" :wandb.Image(embed_activation_fig)}, commit = False)
                wandb.log({"Regression pathway embeddings_" :wandb.Image(reg_activation_fig)}, commit = False)
                
                del embed_activation_fig
                del reg_activation_fig

                gc.collect()

            vis_rho_pred = vis_out_pathway2[:,0,...] #---> (b,13,8)
            vis_delta_z_pred = vis_out_pathway2[:,1,...] #--> (b,13,8)
            vis_cls_score_pred = vis_out_pathway2[:,2,...] # --> (b,13,8)
            # print(vis_cls_score_pred)
            vis_phi_pred = vis_out_pathway2[:,3:,...] # --> (b,10,13,8) ---> (b,13,8)
            
            #TODO: make a separate function for this part of vis later.
            for b in range(vis_rho_pred.shape[0]):
                vis_img_path = vis_batch["image_full_path"][b]
                vis_img = cv2.imread(vis_img_path)
                
                #offset predictions
                vis_rho_pred_b = vis_rho_pred[b,:,:].detach().cpu().numpy()
                vis_phi_pred_b = vis_phi_pred[b,:,:,:].detach().cpu().numpy()
                vis_delta_z_pred_b = vis_delta_z_pred[b,:,:].detach().cpu().numpy()
                vis_cls_score_pred_b = vis_cls_score_pred[b,:,:].detach().cpu().numpy()
                
                #embedding predictions
                vis_embedding_b = vis_out_pathway1[b,:,:,:] # - (4, H, W)
                vis_embedding_b = p(vis_embedding_b) # (4, H/tile_size, W/tile_size)

                vis_embedding_b = vis_embedding_b.detach().cpu().numpy()
                vis_embedding_b = np.transpose(vis_embedding_b, (1,2,0)) # (H/tile_size, W/tile_size, 4)

                vis_cls_score_pred_b[vis_cls_score_pred_b  >= cfg.threshold_score] = 1 # probs to 0 or 1
                vis_cls_score_pred_b[vis_cls_score_pred_b  < cfg.threshold_score] = 0

                #unormalize the rho and delta z
                if cfg.normalize == True:
                    vis_rho_pred_b = vis_rho_pred_b * (cfg.max_lateral_offset - cfg.min_lateral_offset) + cfg.min_lateral_offset
                    vis_delta_z_pred_b = vis_delta_z_pred_b * (cfg.max_delta_z - cfg.min_delta_z) + cfg.min_delta_z
                else: 
                    vis_rho_pred_b = vis_rho_pred_b
                    vis_delta_z_pred_b = vis_delta_z_pred_b

                vis_cam_height_b = vis_batch["vis_gt_height"][b]
                vis_cam_pitch_b = vis_batch["vis_gt_pitch"][b]
                
                #Cluster the tile embedding as per lane class
                # return the tile labels: 0 marked as no lane
                clustered_tiles = embedding_post_process(vis_embedding_b, vis_cls_score_pred_b) 
                print("check if the num of lanes::",np.unique(clustered_tiles))
                
                #extract points from predictions
                points = [] ## ---> [[points lane1 (lists)], [points lane2(lists))], ...]
                for i, lane_idx in enumerate(np.unique(clustered_tiles)): #must loop as the number of lanes present in the scene, max == 5
                    if lane_idx == 0: #no lane ::ignored
                        continue
                    curr_idx = np.where(clustered_tiles == lane_idx) # --> tuple (rows, comumns) idxs

                    rho_lane_i = vis_rho_pred_b[curr_idx[0], curr_idx[1]]

                    phi_vec_lane_i =vis_phi_pred_b[:,curr_idx[0], curr_idx[1]] # ---> 1d array of 10 elements containing probs 
                    phi_lane_i = [palpha2alpha(phi_vec_lane_i[:,i]) for i in range(phi_vec_lane_i.shape[1])]
                    
                    delta_z_lane_i = vis_delta_z_pred_b[curr_idx[0], curr_idx[1]]

                    points_lane_i = [polar_to_catesian(phi_lane_i[i], vis_cam_pitch_b, vis_cam_height_b, delta_z_lane_i[i], rho_lane_i[i]) for i in range(len(phi_lane_i))]
                    points.append(points_lane_i)  
                
                #list containing arrays of lane points
                #TODO: obtain a single plot with all the plots
                gt_fig = vis.draw_lanes(vis_batch["gt_lane_points"][b], vis_img, vis_batch["vis_gt_height"][b], vis_batch["vis_gt_pitch"][b])         
                
                pred_fig = vis.draw_lanes(points, vis_img, vis_batch["vis_gt_height"][b], vis_batch["vis_gt_pitch"][b])
            
                gt_numpy_fig = mplfig_to_npimage(gt_fig)
                pred_numpy_fig = mplfig_to_npimage(pred_fig)

                vis_gt_numpy_fig = cv2.cvtColor(gt_numpy_fig, cv2.COLOR_BGR2RGB)
                vis_pred_numpy_fig = cv2.cvtColor(pred_numpy_fig, cv2.COLOR_BGR2RGB)

                wandb.log({"validate Predictions":wandb.Image(vis_pred_numpy_fig)}, commit = False)
                wandb.log({"validate GT":wandb.Image(vis_gt_numpy_fig)}, commit = False)
                
                del vis_gt_numpy_fig
                del vis_pred_numpy_fig

                gc.collect()
                #TODO: increase the number of visualization images to be displayed and retain the step at per epoch
                break #visualize only one sample for now per vis iteration
            
            
            break

def validate(model2d, model3d, val_loader, cfg, p, device):
    
    model3d.eval()
    
    print(">>>>>>>Validating<<<<<<<<")
    val_loss = 0.0
    val_batch_loss = 0.0
    pred_file_name =  cfg.train_run_name + 'test_pred_file.json'
    lane_pred_file = os.path.join(cfg.lane_pred_dir, pred_file_name)
    
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            
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
                            "gt_delta_z":val_data[8].to(device),
                            "img_id":val_data[10],
                            "val_gt_height":val_data[1].cpu().numpy(),
                            "val_gt_pitch":val_data[2].cpu().numpy()})

                #update projection
                model3d.update_projection(cfg, val_batch["gt_height"], val_batch["gt_pitch"])

                val_o = model2d(val_batch["input_image"].contiguous().float())
                
                a = torch.argmax(val_o, dim =1)
                print(torch.unique(a))
                
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
                val_phi_pred = val_out_pathway2[:,3:,...]

                val_loss1 = discriminative_loss(p(val_out1["embed_out"]), val_batch["gt_lane_cls"],cfg)
                val_loss2 = classification_regression_loss(L1loss, BCEloss, CEloss, val_rho_pred, val_batch["gt_rho"], val_delta_z_pred, val_batch["gt_delta_z"], val_cls_score_pred, val_batch["gt_cls_score"], val_phi_pred, val_batch["gt_phi"])
            
                val_overall_loss = cfg.w_clustering_Loss * val_loss1 + cfg.w_classification_Loss * val_loss2
                # val_overall_loss = val_loss1 + val_loss2 
                
                val_batch_loss = val_overall_loss.detach().cpu() / cfg.batch_size
                val_loss += val_batch_loss
            
                if (val_itr +1) % 10 == 0:
                    val_running_loss = val_loss.item() / (val_itr + 1)
                    print(f"Validation: {val_itr+1} steps of ~{val_loader_len}.  Validation Running Loss {val_running_loss:.4f}")
                
                for b in range(val_rho_pred.shape[0]):
                    #offset predictions
                    val_rho_pred_b = val_rho_pred[b,:,:].detach().cpu().numpy()
                    val_phi_pred_b = val_phi_pred[b,:,:,:].detach().cpu().numpy()
                    val_delta_z_pred_b = val_delta_z_pred[b,:,:].detach().cpu().numpy()
                    val_cls_score_pred_b = val_cls_score_pred[b,:,:].detach().cpu().numpy()
                    
                    #embedding predictions
                    val_embedding_b = val_out_pathway1[b,:,:,:] # - (4, H, W)
                    val_embedding_b = p(val_embedding_b) # (4, H/tile_size, W/tile_size)

                    val_embedding_b = val_embedding_b.detach().cpu().numpy()
                    val_embedding_b = np.transpose(val_embedding_b, (1,2,0)) # (H/tile_size, W/tile_size, 4)

                    # print(vis_cls_score_pred_b)
                    val_cls_score_pred_b[val_cls_score_pred_b  >= cfg.threshold_score] = 1 # probs to 0 or 1
                    val_cls_score_pred_b[val_cls_score_pred_b  < cfg.threshold_score] = 0
                    
                    #unormalize the rho and delta z
                    if cfg.normalize == True:
                        val_rho_pred_b = val_rho_pred_b * (cfg.max_lateral_offset - cfg.min_lateral_offset) + cfg.min_lateral_offset
                        val_delta_z_pred_b = val_delta_z_pred_b * (cfg.max_delta_z - cfg.min_delta_z) + cfg.min_delta_z
                    else:
                        val_rho_pred_b = val_rho_pred_b
                        val_delta_z_pred_b = val_delta_z_pred_b

                    val_cam_height_b = val_batch["val_gt_height"][b]
                    val_cam_pitch_b = val_batch["val_gt_pitch"][b]
                    
                    #Cluster the tile embedding as per lane class
                    # return the tile labels: 0 marked as no lane
                    val_clustered_tiles = embedding_post_process(val_embedding_b, val_cls_score_pred_b)
                    print("check if the num of lanes::",np.unique(val_clustered_tiles))

                    # extract points from predictions
                    points = [] ## ---> [[points lane1 (lists)], [points lane2(lists))], ...]
                    for i, lane_idx in enumerate(np.unique(val_clustered_tiles)): #must loop as the number of lanes present in the scene, max == 5
                        if lane_idx == 0: #no lane ::ignored
                            continue
                        curr_idx = np.where(val_clustered_tiles == lane_idx) # --> tuple (rows, comumns) idxs

                        rho_lane_i = val_rho_pred_b[curr_idx[0], curr_idx[1]]

                        phi_vec_lane_i =val_phi_pred_b[:,curr_idx[0], curr_idx[1]] # ---> 1d array of 10 elements containing probs 
                        phi_lane_i = [palpha2alpha(phi_vec_lane_i[:,i]) for i in range(phi_vec_lane_i.shape[1])]
                        
                        delta_z_lane_i = val_delta_z_pred_b[curr_idx[0], curr_idx[1]]

                        points_lane_i = [polar_to_catesian(phi_lane_i[i], val_cam_pitch_b, val_cam_height_b, delta_z_lane_i[i], rho_lane_i[i]) for i in range(len(phi_lane_i))]
                        points.append(points_lane_i)
                    
                    #write the lane points per batch to the pred_file for evaluation
                    img_id = val_batch["img_id"][b]
                    json_line =valid_set_labels[img_id]
                    json_line["laneLines"] = points
                    json.dump(json_line, jsonFile)
                    jsonFile.write("\n")
        
            val_avg_loss = val_loss / (val_itr +1)
            print(f"Validation Loss: {val_avg_loss}")
            
        #evaluating the predictions
        eval_stats = evaluator.bench_one_submit(lane_pred_file, gt_file_path)
        
        print("===> Evaluation on validation set: \n"
        "laneline F-measure {:.8} \n"
        "laneline Recall  {:.8} \n"
        "laneline Precision  {:.8} \n"
        "laneline x error (close)  {:.8} m\n"
        "laneline x error (far)  {:.8} m\n"
        "laneline z error (close)  {:.8} m\n"
        "laneline z error (far)  {:.8} m\n\n"
        .format(eval_stats[0], eval_stats[1], eval_stats[2], eval_stats[3],
                eval_stats[4], eval_stats[5], eval_stats[6]))

    return eval_stats, val_avg_loss 

def train(model2d, model3d, train_loader, val_loader, cfg, epoch, optimizer2, scheduler2, L1loss, BCEloss, CEloss, m, p, device,  best_fmeasure, optimizer1 = None, scheduler1 = None):
     #init best measure before the start of the training
    print(model2d)
    # print(model2d)
    batch_loss = 0.0 
    tr_loss = 0.0 
    tr_loss1 = 0.0
    tr_loss2 = 0.0
    start_point = time.time()

    timings = dict()
    multitimings = MultiTiming(timings)
    multitimings.start('batch_load')

    if args.e2e == True:
        #TOOD: solve this issue of missing outs when both are .train()
        model3d.train()
    else:
        model3d.train()

    for itr, data in enumerate(train_loader):
        
        batch_load_time = multitimings.end('batch_load')
        print(f"Got new batch: {batch_load_time:.2f}s - training iteration: {itr}")

        #flag for train log and validation loop
        ## TODO: change it to per epoch not iteration
        
        should_log_train = (itr+1) % cfg.train_log_frequency == 0 
        should_run_valid = (itr+1) % cfg.val_frequency == 0
        should_run_vis = (itr+1) % cfg.vis_frequency == 0

        multitimings.start('train_batch')

        #get the data
        batch = {}
        with Timing(timings, "inputs_to_GPU"):
            batch.update({"input_image":data[0].to(device),
                        "aug_mat":data[1].to(device).float(),
                        "gt_height":data[2].to(device),
                        "gt_pitch":data[3].to(device),
                        "gt_lane_points":data[4],
                        "gt_rho":data[5].to(device),
                        "gt_phi":data[6].to(device).float(),
                        "gt_cls_score":data[7].to(device),
                        "gt_lane_cls":data[8].to(device),
                        "gt_delta_z":data[9].to(device),
                        'img_full_path':data[10]
                        })

        #TODO: add the condition for camera fix
        #update projection
        model3d.update_projection(cfg, batch["gt_height"], batch["gt_pitch"])

        # #update augmented matrix
        model3d.update_projection_for_data_aug(batch["aug_mat"])

        optimizer2.zero_grad(set_to_none= True)
        
        with Timing(timings, "2d_forward_pass"):
            #forward pass
            o = model2d(batch["input_image"].float())
            
            
            # ######################## To check for the inference from the model while it trains
    
            # images = []
            # mapping = {(0, 0, 0): 0, (255, 255, 255): 1}
            # rev_mapping = {mapping[k]: k for k in mapping}
            # for i in range(cfg.batch_size):

            #     pred_mask_i = o[i, :, :, :] #--- > (2,h,W)
                
            #     pred_mask_i = torch.argmax(pred_mask_i,0) #--- > (h,W)
                
            #     pred_image = torch.zeros(3,pred_mask_i.size(0), pred_mask_i.size(1), dtype = torch.uint8)

            #     for k in rev_mapping:
            #         pred_image[:,pred_mask_i == k] = torch.tensor(rev_mapping[k]).byte().view(3,1)
            #         pred_img = pred_image.permute(1,2,0).numpy()
                    
            #     org_image = cv2.imread(batch['img_full_path'][i]) 
                
            #     print("checking the sghape og original image",org_image.shape)

            #     pred_img = cv2.resize(pred_img, (org_image.shape[1], org_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            #     vis_img = cv2.addWeighted(org_image,0.5, pred_img,0.5,0)

            #     vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            #     # print(pred_img)
            #     image_name = 'check_infer_' + str(itr) + '.jpg'
            #     image_save_path = os.path.join('/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/infer',image_name)
            #     cv2.imwrite(image_save_path, vis_img)
            #     # images.append(vis_img)
                ######################################################################
                #         
            print("checking if model 2d correct in training")
            a = torch.argmax(o, dim =1)

            print(torch.unique(a))            
            o = o.softmax(dim=1)
            o = o/torch.max(torch.max(o, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] 
            # print("shape of o before max", o.shape)
            o = o[:,1:,:,:]

        with Timing(timings, "3d_forward_pass"):
            out1 = model3d(o)

        out_pathway1 = out1["embed_out"]
        out_pathway2 = out1["bev_out"]

        rho_pred = out_pathway2[:,0,...]
        delta_z_pred = out_pathway2[:,1,...]
        cls_score_pred = out_pathway2[:,2,...]
        phi_pred = out_pathway2[:,3:,...]

        with Timing(timings, "3d_Lane_loss_calculation"):
            loss1 = discriminative_loss(p(out1["embed_out"]), batch["gt_lane_cls"],cfg)
            loss2 = classification_regression_loss(L1loss, BCEloss, CEloss, rho_pred, batch["gt_rho"], delta_z_pred, batch["gt_delta_z"], cls_score_pred, batch["gt_cls_score"], phi_pred, batch["gt_phi"])

            print("==>discriminative loss::", loss1.item())
            print("==>classification loss::", loss2.item())

            if cfg.weighted_loss:
                overall_loss = cfg.w_clustering_Loss * loss1 + cfg.w_classification_Loss * loss2
            else:
                overall_loss = loss1 + loss2

        with Timing(timings, "backward_pass"):                     
            if cfg.fix_branch and epoch < cfg.fix_branch_epoch :
                print("Training only Embeddings first ====>")
                freeze_network(model3d, "bev_encoder")
                overall_loss.backward()
            else: 
                print("Training regression branch ====>")
                freeze_network(model3d, "embedding")
                overall_loss.backward()
        
        with Timing(timings, 'clip_gradients'):
            torch.nn.utils.clip_grad_norm_(model3d.parameters(), cfg.grad_clip)
        
        with Timing(timings,  "optimizer_step"): 
            optimizer2.step()

        batch_loss1 = loss1.detach().cpu()/cfg.batch_size
        batch_loss2 = loss2.detach().cpu()/cfg.batch_size
        batch_loss = overall_loss.detach().cpu() / cfg.batch_size 
        train_batch_time= multitimings.end('train_batch')

        #reporting model fps
        fps = cfg.batch_size / train_batch_time
        print(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")
        
        tr_loss1 += batch_loss1
        tr_loss2 += batch_loss2
        tr_loss += batch_loss

        # eval loop
        if should_run_valid:
            with Timing(timings, "validate loop"):
                eval_stats, val_avg_loss = validate(model2d, model3d, val_loader, cfg, p, device) 

                    #save the best model
                if eval_stats[0] > best_fmeasure:
                    best_fmeasure = eval_stats[0]

                    #TODO: alter this model checkpointing: Trained e2e
                    print(">>>>>>> Creating model Checkpoint <<<<<<<")
                    checkpoint_file_name = cfg.train_run_name + args.data_split + str(val_avg_loss.item()) + "epoch_" + str(epoch+1) + ".pth"
                    checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_file_name)
                    torch.save(model3d.state_dict(), checkpoint_save_path)
                
            wandb.log({'Validation_loss': val_avg_loss}, commit = False)
            scheduler2.step(val_avg_loss.item())  
            #TODO: add the condition for e2e
            model3d.train()              
            
        #vis loop
        if should_run_vis:
            with Timing(timings, "visualize predictions and ground truth"):
                visualization(cfg, model2d, model3d, vis_loader, p, device, epoch, itr)
            model3d.train()

        if should_log_train:
            running_loss1 = tr_loss1.item()/ cfg.train_log_frequency
            running_loss2 = tr_loss2.item()/cfg.train_log_frequency
            running_loss = tr_loss.item() / cfg.train_log_frequency
            print(f"Epoch: {epoch+1}/{cfg.epochs}. Done {itr+1} steps of ~{train_loader_len}. Running Loss:{running_loss:.4f}")
            pprint_stats(timings)

            wandb.log({'epoch': epoch, 
                    'discrminative_loss': running_loss1,
                    'class_reg_loss': running_loss2,
                    'train_loss':running_loss,
                    'lr': scheduler2.optimizer.param_groups[0]['lr'],
                    **{f'time_{k}': v['time'] / v['count'] for k, v in timings.items()}
                    }, commit=True)
            """"
            #TODO: remove it later just put here to test the intial training, once the loader is fast remove it and test it again.
            """
            tr_loss  = 0.0
            tr_loss1 = 0.0
            tr_loss2 = 0.0

    #reporting epoch train time 
    print(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time()- start_point)}")
    return best_fmeasure

def image_to_tensor(img):
    img_mean = [0.485, 0.456, 0.406] 
    img_std = [0.229, 0.224, 0.225]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img[160:,:,:]
    img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
    
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean= img_mean, std=img_std)(img)

    return img

def freeze_network(model, layer):
    for name, p in model.named_parameters():
        # freeze the regression layers
        if layer in name:
            p.requires_grad = False

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
    parser.add_argument("--data_dir", type=str, default="/home/ims-robotics/Documents/gautam/dataset/Apollo_Sim_3D_Lane_Release", help="data directory")
    parser.add_argument("--data_split", type=str, default="standard", help="data split")
    parser.add_argument("--path_data_split", type=str, default="/home/ims-robotics/Documents/gautam/dataset/data_splits", help="path to data split")
    parser.add_argument("--e2e",type=bool, default=False, help="enable end-to-end training")
    
    #parsing args
    args = parser.parse_args()

    #load config file
    cfg = Config.fromfile(args.config)

    #wandb init 
    run = wandb.init(entity = os.environ["WANDB_ENTITY"], project = os.environ["WANDB_PROJECT"], name = cfg.train_run_name, mode = 'offline' if args.no_wandb else 'online')

    # for reproducibility
    torch.manual_seed(args.seed)
    # np.random.seed(args.seed) 
    # random.seed(args.seed)

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
    
    gt_file_path = os.path.join(data_split, 'test.json')
    
    #extract valid set labels for eval
    global valid_set_labels
    valid_set_labels = [json.loads(line) for line in open(gt_file_path).readlines()]
    
    #initialise the evaluator
    evaluator = Apollo_3d_eval.LaneEval(cfg)
    
    train_dataset = Apollo3d_loader(data_root, data_split, shuffle = True, cfg = cfg, phase = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=cfg.batch_size, collate_fn= None, prefetch_factor=2, persistent_workers=True, worker_init_fn= configure_worker)
    train_loader = BatchDataLoader(train_loader, batch_size = cfg.batch_size, mode ='train')
    train_loader_len = len(train_loader)
    train_loader = BackgroundGenerator(train_loader)
    
    val_dataset = Apollo3d_loader(data_root, data_split, shuffle = False, cfg = cfg, phase = 'test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=cfg.batch_size, collate_fn= None, prefetch_factor=2, persistent_workers=True, worker_init_fn= configure_worker)
    val_loader = BatchDataLoader(val_loader, batch_size = cfg.batch_size, mode = 'test')
    val_loader_len = len(val_loader)
    val_loader = BackgroundGenerator(val_loader)

    vis_dataset = Apollo3d_loader(data_root, data_split, shuffle = False, cfg = cfg, phase = 'test')
    vis_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=cfg.batch_size, collate_fn= None, prefetch_factor=2, persistent_workers=False, worker_init_fn= configure_worker)
    vis_loader = BatchDataLoader(vis_loader, batch_size = cfg.batch_size, mode = 'test')
    vis_loader_len = len(vis_loader)
    vis_loader = BackgroundGenerator(vis_loader)
    
    print("===> batches in train loader", train_loader_len)
    print("===> batches in val loader", val_loader_len)
    
    #load model and weights
    if args.e2e == True: #TODO: make a single forward function for the combined model
        #load 2d model from checkpoint and train the whole pipeline end-to-end
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device)
        wandb.watch(model2d)
        wandb.watch(model3d)
    else: 
        model2d = load_model(cfg, baseline=args.baseline, pretrained = args.pretrained2d).to(device) #args.pretrained2d == TRUE
        model3d = load_3d_model(cfg, device, pretrained=args.pretrained3d).to(device)
        print(model3d)
        wandb.watch(model3d)

    #TODO: remove it later when e2e 
    model2d.train()
    

    # ################# to Enable to check the inference of the trained 2d model ##################
    # # # image_tusimple = cv2.imread("/home/gautam/Thesis/E2E_3DLane_AuxNet/vis_test/8.jpg")
    # # image_apollo = cv2.imread("/home/ims-robotics/Documents/gautam/E2E_3DLane_AuxNet/dog.jpg")
    
    # # image = image_to_tensor(image_apollo)
    
    # # image = image.unsqueeze(0)
    # # image = image.float().to(device)
    # # # TODO: test here

    # # out = model2d(image)

    # # preds = torch.argmax(out[0,:,:,:], 0)
    # # print(preds)
    # # print(torch.unique(preds))

    # # #  print(preds.shape)
    # # # print(torch.unique(preds))
    # # #save the binary predicted mask using opencv 
    # # mapping = {(0, 0, 0): 0, (255, 255, 255): 1}
    # # rev_mapping = {mapping[k]: k for k in mapping}
    # # # for i in range(cfg.batch_size):

    # # # pred_mask_i = preds[0, :, :, :] #--- > (2,h,W)
    
    # # # pred_mask_i = torch.argmax(pred_mask_i,0) #--- > (h,W)
    
    # # pred_image = torch.zeros(3,preds.size(0), preds.size(1), dtype = torch.uint8)
    
    # # for k in rev_mapping:
    # #     pred_image[:,preds == k] = torch.tensor(rev_mapping[k]).byte().view(3,1)
    # #     pred_img = pred_image.permute(1,2,0).numpy()
        
    # # pred_img = cv2.resize(pred_img, (1920, 1080), interpolation=cv2.INTER_NEAREST)

    # # vis_img = cv2.addWeighted(image_apollo,0.5, pred_img,0.5,0)
    # # cv2.imwrite("inferapollo.jpg", vis_img)
    # #########################################################################################

    #general loss functions
    L1loss= nn.L1Loss().to(device)
    #NOTE: verify that BCEWithLogitsLoss for score as We know that when the last layer has acitvations normal loss can be used (numerically stable rn) 
    BCEloss = nn.BCEWithLogitsLoss().to(device)
    CEloss = nn.CrossEntropyLoss().to(device)
    m = nn.Sigmoid()
    #TODO: Chnage the by selected tile_size in the end
    p = nn.MaxPool2d(cfg.tile_size,stride = cfg.tile_size)

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
    
    #train_loop
    best_fmeasure = 0.0
    print("======> Starting to train")
    with run:
        print("==> Reporting Argparse params to wandb")
        for arg in vars(args):
            wandb.config.update({arg: getattr(args, arg)})
            print(arg, getattr(args, arg))
        
        print("==> Reporting config params to wandb")
        for arg1 in vars(cfg):
            print(arg, getattr(cfg, arg1))
        
        #for speedup
        with torch.autograd.profiler.profile(enabled=False):
            with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=False):
                for epoch in tqdm(range(cfg.epochs)):

                    if args.pretrained2d == True:
                        M = train(model2d, model3d, train_loader, val_loader, cfg, epoch, optimizer2, scheduler2, L1loss, BCEloss, CEloss, m, p, device, best_fmeasure)
                    else:
                        M = train(model2d, model3d, train_loader, val_loader, cfg, epoch, optimizer2, scheduler2, L1loss, BCEloss, CEloss, m, p, device, optimizer1, scheduler1, best_fmeasure)

                        best_fmeasure = M
                train_2d_model_save_path = os.path.join(result_model_dir, cfg.train_run_name + "2d.pth")
                train_3d_model_save_path = os.path.join(result_model_dir, cfg.train_run_name + "3d.pth")

                if args.pretrained2d == True:
                    torch.save(model3d.state_dict(), train_3d_model_save_path)
                    print("=> Saving 3d model to", train_3d_model_save_path)
                else: 
                    torch.save(model2d.state_dict(), train_2d_model_save_path)
                    torch.save(model3d.state_dict(), train_3d_model_save_path)
                    print("==> Saved the trained models to:", train_2d_model_save_path, train_3d_model_save_path)
                print("==>Training Finished")
