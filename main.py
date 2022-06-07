#TODO: Change the name of the script to 2D_lane_detection_train.py

from contextlib import redirect_stderr
from pprint import pprint
import torch 
import cv2 
from tqdm import tqdm
import os 
import dotenv
dotenv.load_dotenv()
import wandb
import time

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

def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

def visualizaton(model, device, data_loader,cfg):
    ##TODO: add the visualization for overlays and gt also

    model.eval()
    print(">>>>>>>visualization<<<<<<")
    with torch.no_grad():
        for vis_itr, vis_data in enumerate(data_loader): 
            # batch size is suppose 6, therfore there exists 6 images only that needs to be vis and break after that 
            
            vis_image = vis_data['img'].to(device) #to be used for predictions 
            orig_image_path = vis_data['full_img_path']

            predictions = {}
            vis_pred = model(vis_image)
            
            images = []
            if cfg.train_type == "multi_class":
                for i in range(cfg.batch_size):

                    predictions.update({"seg":vis_pred[i:i+1, :, :, :] })
                    
                    vis_img = vis.draw_lines(orig_image_path[i], predictions)
                    # vis_image_path = "/home/gautam/Thesis/E2E_3DLane_AuxNet/vis_test/test" + str(i) + ".jpg"
                    
                    images.append(vis_img)

                wandb.log({"validate Predictions":[wandb.Image(image) for image in images]})
                break
            
            elif cfg.train_type == "binary":
                
                images = []
                mapping = {(0, 0, 0): 0, (255, 255, 255): 1}
                rev_mapping = {mapping[k]: k for k in mapping}
                for i in range(cfg.batch_size):

                    pred_mask_i = vis_pred[i, :, :, :] #--- > (2,h,W)
                    
                    pred_mask_i = torch.argmax(pred_mask_i,0) #--- > (h,W)
                    
                    pred_image = torch.zeros(3,pred_mask_i.size(0), pred_mask_i.size(1), dtype = torch.uint8)

                    for k in rev_mapping:
                        pred_image[:,pred_mask_i == k] = torch.tensor(rev_mapping[k]).byte().view(3,1)
                        pred_img = pred_image.permute(1,2,0).numpy()
                        
                    pred_img = cv2.resize(pred_img, (cfg.ori_img_w, cfg.ori_img_h - cfg.cut_height), interpolation=cv2.INTER_NEAREST)
                    
                    org_image = cv2.imread(orig_image_path[i])
                    org_image = org_image[cfg.cut_height:, :, :]

                    vis_img = cv2.addWeighted(org_image,0.5, pred_img,0.5,0)

                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    images.append(vis_img)
                    
                wandb.log({"validate Predictions":[wandb.Image(image) for image in images]})
                break

def validate(model, device, data_loader, loss_f, cfg):
    model.eval()         
    print(">>>>>>>>>Validating<<<<<<<<<")
    
    val_loss = 0.0  
    val_batch_loss = 0.0
    with torch.no_grad():
        val_pred = []
        pred_out = {}
        Iou_batch_list = []

        for val_itr, val_data in enumerate(data_loader):
            
            val_gt_mask = val_data['binary_mask'].to(device).long()
            val_img = val_data['img'].to(device)
            
            val_seg_out = model(val_img)
            pred_out.update({'seg':val_seg_out})
            
            if cfg.train_type == "multi_class":
                pred_out_list = vis.get_lanes(pred_out)
                val_pred.extend(pred_out_list)

            else:
                #calcualte IOU
                metric_batch = iou(torch.argmax(val_seg_out,1), val_gt_mask)
                Iou_batch_list.append(metric_batch)
            
            # val_seg_loss = loss_f(torch.max(val_seg_out, dim =1)[0], val_gt_mask)
            val_seg_loss = loss_f(val_seg_out, val_gt_mask)

            #NOTE: To be used when not using binary segmentation
            # val_seg_loss = loss_f(F.log_softmax(val_seg_out, dim =1), val_gt_mask.long())
            val_batch_loss = val_seg_loss.detach().cpu() / cfg.batch_size

            val_loss += val_batch_loss

            if (val_itr+1) % 10 == 0:
                val_running_loss = val_loss.item() / (val_itr+1)
                print(f"Validation: {val_itr+1} steps of ~{val_loader_len}.  Validation Running Loss {val_running_loss:.4f}")    

        val_data_IOU = torch.mean(torch.stack(Iou_batch_list))

        val_avg_loss = val_loss / (val_itr+1)
        print(f"Validation Loss: {val_avg_loss}")

    return val_avg_loss, val_pred, val_data_IOU
   

def train(model, device, train_loader, val_loader, scheduler, optimizer, epoch, cfg, criterion):
    metric = 0
    Iou = 0.0
    batch_loss = 0.0
    tr_loss = 0.0
    start_point = time.time()
    
    timings = dict()
    multitimings = MultiTiming(timings)
    multitimings.start('batch_load')

    for itr, data in enumerate(train_loader): 
        model.train()

        #TODO: correct batch_loading time 
        batch_load_time = multitimings.end('batch_load')
        print(f"Got new batch: {batch_load_time:.2f}s - training iteration: {itr}")
    
        #flag for train log and validation loop
        should_log_train = (itr+1) % cfg.train_log_frequency == 0 
        should_run_valid = (itr+1) % cfg.val_frequency == 0
        should_run_vis = (itr+1) % cfg.val_frequency == 0

        multitimings.start('train_batch')

        optimizer.zero_grad(set_to_none=True)
        
        with Timing(timings, "inputs_to_GPU"):
            gt_mask = data['binary_mask'].to(device).long()
            input_img = data['img'].to(device)
            
        with Timing(timings,"forward_pass"):
            seg_out = model(input_img)

        with Timing(timings,"seg_loss"):
            
            #NOTE: to use in the case if not binary segmentation
            # seg_loss = criterion(F.log_softmax(seg_out, dim =1), gt_mask.long())
            
            #NOTE: to use in the case if binary segmentation with BCElosswithlogits
            # seg_loss = criterion(torch.max(seg_out, dim =1)[0],gt_mask)
            seg_loss = criterion(seg_out, gt_mask)

            print("seg_loss_train: ", seg_loss)
            # TODO: add a condition of lane exist loss
        
        with Timing(timings,"backward_pass"):    
            seg_loss.backward()

        #TODO: Add clippping gradients with argparse
        with Timing(timings, "optimizer step"):
            optimizer.step()
        
        batch_loss = seg_loss.detach().cpu() * cfg.batch_size
        train_batch_time= multitimings.end('train_batch')
        
        #reporting model FPS
        fps = cfg.batch_size / train_batch_time
        print(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")

        tr_loss += batch_loss

        if should_log_train: 

            running_loss = tr_loss.item() / cfg.train_log_frequency 
            print(f"Epoch: {epoch+1}/{cfg.epochs}. Done {itr+1} steps of ~{train_loader_len}. Running Loss:{running_loss:.4f}")
            pprint_stats(timings)

            wandb.log({'epoch': epoch, 
                        'train_loss':running_loss,
                        'lr': scheduler.optimizer.param_groups[0]['lr'],
                        **{f'time_{k}': v['time'] / v['count'] for k, v in timings.items()}
                        }, commit=True)

            tr_loss = 0.0

        # eval Loop 
        if should_run_valid:
            with Timing(timings, "validate"):
                
                #TODO: Later add more metrics for binary segmentation
                val_avg_loss, val_pred, Iou_val = validate(model, device, val_loader, criterion, cfg)
                
                if cfg.train_type == "multi_class":
                # evaluate 
                    pred_json_save_path = "/home/gautam/Thesis/E2E_3DLane_AuxNet/pred_json" 
                    
                    curr_metric = val_loader.dataset.evaluate(val_pred, pred_json_save_path)
                    curr_acc = curr_metric["Accuracy"]
                    print("Current Acccuracy",curr_acc)
                    
                    #making model checkpoints on the basis of accuracy
                    if curr_acc> metric:
                        metric = curr_acc
                        print(f"Best Metric for Epoch:{epoch+1} and train itr. {itr} is: {curr_metric} ")

                        print(">>>>>>>>Creating model Checkpoint<<<<<<<")
                        checkpoint_save_file = cfg.train_run_name + str(val_avg_loss.item()) +"_" + str(epoch+1) + ".pth"
                        checkpoint_save_path = os.path.join(checkpoints_dir,checkpoint_save_file)

                        torch.save(model.state_dict(),checkpoint_save_path)
                
                elif cfg.train_type == "binary":
                    
                    if Iou_val > Iou:
                        Iou = Iou_val
                        print(f"Best IOU for Epoch:{epoch+1} and train itr. {itr} is: {Iou_val} ")

                        print(">>>>>>>>Creating model Checkpoint<<<<<<<")
                        checkpoint_save_file = cfg.train_run_name + str(val_avg_loss.item()) +"_" + str(epoch+1) + ".pth"
                        checkpoint_save_path = os.path.join(checkpoints_dir,checkpoint_save_file)

                        torch.save(model.state_dict(),checkpoint_save_path)
        
            wandb.log({'Validation_loss': val_avg_loss,}, commit=False)
            
            scheduler.step(val_avg_loss.item())

        if should_run_vis: 
            with Timing(timings,"Visualising predictions"):
                visualizaton(model, device, val_loader,cfg)

    #reporting epoch train time 
    print(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time()- start_point)}")

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
    checkpoints_dir = './nets/checkpoints'
    result_model_dir = './nets/model_itr'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    # dataloader
    train_loader = build_dataloader(cfg.dataset.train, cfg, is_train = True)
    val_loader = build_dataloader(cfg.dataset.val, cfg, is_train = False)

    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)
    
    print("===> batches in train loader", train_loader_len)
    print("===> batches in val loader", val_loader_len)
    
    model = load_model(cfg, baseline = args.baseline)
    model = model.to(device)
    
    wandb.watch(model)

    #segmentation loss
    """
        posweight is calculated as:
        fraction of positve samples can be calculated as: target_mask.mean()
        posweight = 1 - fraction of 1s/ fraction of 1          
        """
    if cfg.train_type == "binary":
            
        #NOTE: TO Use when binary segmentation and the output from the model has no activations and is just a single channel
        pos_weight = torch.tensor([1.0,19.0]).to(device) #basically it means that I have 19 positive samples and 1 negative sample
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight = pos_weight).to(device)

    else:
        # NOTE: TO use when multi class segmentation is done.
        criterion = torch.nn.NLLLoss().to(device)

    #optimizer and scheduler
    param_group = model.parameters()
    optimizer = topt.Adam(param_group, cfg.lr, weight_decay= cfg.l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor= cfg.lrs_factor, patience= cfg.lrs_patience,
                                                        threshold= cfg.lrs_thresh, verbose=True, min_lr= cfg.lrs_min,
                                                        cooldown=cfg.lrs_cd)                               
    
    with run:
        print("==> Reporting Argparse params")
        for arg in vars(args):
            wandb.config.update({arg: getattr(args, arg)})
            print(arg, getattr(args, arg))
        
        #for speedup
        with torch.autograd.profiler.profile(enabled=False):
            with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=False):
                for epoch in tqdm(range(cfg.epochs)):                    
                    train(model, device, train_loader, val_loader, scheduler, optimizer, epoch, cfg, criterion)
    
                train_model_savepath = os.path.join(result_model_dir, cfg.train_run_name + ".pth")
                torch.save(model.state_dict(), train_model_savepath)
                print("Saved the train model")
                print("Training finished")