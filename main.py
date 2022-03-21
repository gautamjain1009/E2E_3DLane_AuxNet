from pprint import pprint
import torch 
import cv2 
from tqdm import tqdm
import os 
import dotenv
import wandb
import time

from utils.config import Config
import argparse 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torch.nn.functional as F
import torch.optim as topt

from .timing import *
import logging 
logging.basicConfig(level = logging.DEBUG)

#build import for different modules
from datasets.registry import build_dataloader
from models.registry import build_baseline


def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"

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
    parser.add_argument("--date_it", type=str, required=True, help="run date/name")  # "16Jan_1_seg"
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--seed", type=int, default=27, help="random seed")

    #wandb init
    # run = wandb.init(entity = os.environ["WANDB_ENTITY"], project = os.environ["WANDB_PROJECT"], name = cfg.train_run_name, model = 'offline' if args.no_wandb else 'online') 

    #parsing args
    args = parser.parse_args()

    #parasing config file
    cfg = Config.fromfile(args.config)

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
    
    #TODO: add vis laoder

    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)
    
    print("===> batches in train loader", train_loader_len)
    print|("===> batches in val loader", val_loader_len)
    
    #TODO:
        #loss functions (Done)
        # scheduler and optimizer (Done) 
        #idea for test loop and how the evaluation is carried out (Done)
        #trian loop /Eval loop (Done)
        #add timmings
        #Integrate with wandb including model visulization 
        #checkpoint dirs
        #model.train() and model.eval()

    """ model defination TODO: add to build function"""
    def reinitialize_weights(layer_weight):
        torch.nn.init.xavier_uniform_(layer_weight)

    model = build_baseline(cfg.net, cfg)
    model = model.to(device)
    
    #wandb.watch(model)
    # print("before",model.lane_exist.linear1.bias)
    
    #TODO: enable it only for baseline 
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

    # print("After", model.lane_exist.linear1.bias)
    
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
    #for speedup
    # with torch.autograd.profiler.profile(enabled=False):
    #     with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=False):
    for epoch in tqdm(range(cfg.epochs)):
        
        batch_loss = 0.0
        tr_loss = 0.0
        start_point = time.time()
        
        timings = dict()
        multitimings = MultiTiming(timings)
        multitimings.start('batch_load')

        for itr, data in enumerate(train_loader): 
            # model.train()

            batch_load_time = multitimings.end('batch_load')
            print(f"Got new batch: {batch_load_time:.2f}s - training iteration: {itr}")
           
            #flag for train log and validation loop
            should_log_train = (itr+1) % cfg.train_log_frequency == 0 
            should_run_valid = (itr+1) % cfg.val_frequency == 0
            
            multitimings.start('train_batch')
            optimizer.zero_grad(set_to_none=True)

            gt_mask = data['mask'].to(device)
            # print(gt_mask.shape)
            input_img = data['img'].to(device)
            
            seg_out = model(input_img)
            #TODO: verify the dim of the softmax dim
            seg_loss = criterion(F.log_softmax(seg_out, dim =1), gt_mask.long())
    
            # TODO: add a condition of lane exist loss
            seg_loss.backward()

            #TODO: Add clippping gradients with argparse
            optimizer.step()
            
            batch_loss = seg_loss.detach().cpu()/cfg.batch_size
            train_batch_time= multitimings.end('train_batch')
            
            #reporting model FPS
            fps = cfg.batch_size / train_batch_time
            print(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")

            tr_loss += batch_loss

            if should_log_train: 
                
                #TODO: Add timings
                running_loss = tr_loss.item() / cfg.train_log_frequency 
                print(f"Epoch: {epoch+1}/{cfg.epochs}. Done {itr+1} steps of ~{train_loader_len}. Running Loss:{running_loss:.4f}")
                
                #TODO: Add wandb logging

                tr_loss = 0.0

            #eval Loop 
            if should_run_valid:
                #model.eval()
                            
                print(">>>>>>>>>Validating<<<<<<<<<")
                #TODO: add wandb visualization

                val_loss = 0.0  
                val_batch_loss = 0.0
                with Timing(timings, "validate"):
                    with torch.no_grad():
                        for val_itr, val_data in enumerate(val_loader):

                            val_gt_mask = val_data['mask'].to(device)
                            val_img = val_data['img'].to(device)

                        val_seg_loss = model(val_img)
                        
                        val_batch_loss = val_seg_loss.detach().cpu()/cfg.batch_size

                        val_loss += val_batch_loss

                        if (val_itr+1) % 10 == 0:
                            val_running_loss = val_loss.item() / (val_itr+1)
                            print(f"Validation: {val_itr+1} steps of ~{val_loader_len}.  Validation Running Loss {val_running_loss:.4f}")    
                            
                        val_avg_loss = val_loss / (val_itr+1)
                        print(f"Validation Loss: {val_avg_loss}")

                scheduler.step(val_avg_loss.item())

                print(">>>>>>>>Creating model Checkpoint<<<<<<<")
                checkpoint_save_file = "2dLane" + args.date_it + str(val_avg_loss) + str(epoch+1) + ".pth"
                checkpoint_save_path = os.path.join(checkpoints_dir,checkpoint_save_file)

                torch.save(model.state_dict(),checkpoint_save_path)

                #TODO: log wandb validation loss
        
        #reporting epoch train time 
        print(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time()- start_point)}")


    train_model_savepath = os.path.join(result_model_dir, cfg.train_run_name + ".pth")
    torch.save(model.state_dict(), train_model_savepath)
    print("Saved the train model")
    print("Training finished")

                        









                        



            















