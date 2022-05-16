#TODO: Change the name of the script to 2D_lane_detection_train.py

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

def visualizaton(model, device, data_loader):
    model.eval()
    print(">>>>>>>visualization<<<<<<")
    with torch.no_grad():
        for vis_itr, vis_data in enumerate(data_loader): 
            # batch size is suppose 6, therfore there exists 6 images only that needs to be vis and break after that 
            
            vis_image = vis_data['img'].to(device) #to be used for predictions 
            orig_image_path = vis_data['full_img_path']

            predictions = {}
            vis_pred = model(vis_image)
            # print(vis_pred[0:1, :, :, :].shape)
            images = []
            for i in range(cfg.batch_size):

                predictions.update({"seg":vis_pred[i:i+1, :, :, :] })
                
                vis_img = vis.draw_lines(orig_image_path[i], predictions)
                # vis_image_path = "/home/gautam/Thesis/E2E_3DLane_AuxNet/vis_test/test" + str(i) + ".jpg"
                
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

        for val_itr, val_data in enumerate(data_loader):
            
            val_gt_mask = val_data['mask'].to(device)
            val_img = val_data['img'].to(device)
            
            val_seg_out = model(val_img)
            pred_out.update({'seg':val_seg_out})
            pred_out_list = vis.get_lanes(pred_out)
            
            val_pred.extend(pred_out_list)
            
            val_seg_loss = loss_f(F.log_softmax(val_seg_out, dim =1), val_gt_mask.long())

            val_batch_loss = val_seg_loss.detach().cpu()/cfg.batch_size

            val_loss += val_batch_loss

            if (val_itr+1) % 10 == 0:
                val_running_loss = val_loss.item() / (val_itr+1)
                print(f"Validation: {val_itr+1} steps of ~{val_loader_len}.  Validation Running Loss {val_running_loss:.4f}")    
            
        val_avg_loss = val_loss / (val_itr+1)
        print(f"Validation Loss: {val_avg_loss}")

    return val_avg_loss, val_pred

def train(model, device, train_loader, val_loader, scheduler, optimizer, epoch, cfg, criterion):
    metric = 0 
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
            gt_mask = data['mask'].to(device)
            # print(gt_mask.shape)
            input_img = data['img'].to(device)
            
        with Timing(timings,"forward_pass"):
            seg_out = model(input_img)
        
        with Timing(timings,"seg_loss"):
            
            #TODO: verify the dim of the softmax dim and correct
            seg_loss = criterion(F.log_softmax(seg_out, dim =1), gt_mask.long())
            # TODO: add a condition of lane exist loss
        
        with Timing(timings,"backward_pass"):    
            seg_loss.backward()

        #TODO: Add clippping gradients with argparse
        with Timing(timings, "optimizer step"):
            optimizer.step()
        
        batch_loss = seg_loss.detach().cpu()/cfg.batch_size
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

        #eval Loop 
        if should_run_valid:
            with Timing(timings, "validate"):
                val_avg_loss, val_pred = validate(model, device, val_loader, criterion, cfg)
                
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

            wandb.log({'Validation_loss': val_avg_loss,}, commit=False)
            
            scheduler.step(val_avg_loss.item())

        if should_run_vis: 
            with Timing(timings,"Visualising predictions"):
                visualizaton(model, device, val_loader)

    #reporting epoch train time 
    print(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time()- start_point)}")

##Main TODO: remove out['seg'] dependability of the pipieline 
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
    # # print(model)
    # a =torch.rand(1,3,512,512)
    # a = a.to(device)
    # out = model(a)
    # print(out.shape)

    wandb.watch(model)

    #segmentation loss
    criterion = torch.nn.NLLLoss().to(device)
    criterion_exist = torch.nn.BCEWithLogitsLoss().to(device)

    #optimizer and scheduler
    param_group = model.parameters()
    optimizer = topt.Adam(param_group, cfg.lr, weight_decay= cfg.l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor= cfg.lrs_factor, patience= cfg.lrs_patience,
                                                        threshold= cfg.lrs_thresh, verbose=True, min_lr= cfg.lrs_min,
                                                        cooldown=cfg.lrs_cd)                               
    
    #TODO: Add a condition for model to be loaded from pretrained model if needed only for inference
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