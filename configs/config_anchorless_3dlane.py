import numpy as np 

# #TODO: Optimize this config
# """
# Dataloader params
# """

# #TODO: Mya be the mean and std values are not correct for the dataset
# img_norm = dict(
#     mean=[103.939, 116.779, 123.68],
#     std=[1., 1., 1.]
# )

img_height = 360
img_width = 480
cut_height = 160
ori_img_h = 720
ori_img_w = 1280

# #for vis
# sample_y = range(710,150, -10)
# thr = 0.6

# train_augmentation = [
#     dict(type='RandomRotation'),
#     dict(type='RandomHorizontalFlip'),
#     dict(type='Resize', size=(img_width, img_height)),
#     dict(type='Normalize', img_norm=img_norm),
#     dict(type='ToTensor'),
# ] 

# val_augmentation = [
#     dict(type='Resize', size=(img_width, img_height)),
#     dict(type='Normalize', img_norm=img_norm),
#     dict(type='ToTensor')
# ] 

# dataset_path = '/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple'

# dataset = dict(
#     train=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='trainval',
#         transform = train_augmentation
#     ),
#     val=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='test',
#         transform = val_augmentation
#     ),
#     test=dict(
#         type='TusimpleLoader',
#         data_root=dataset_path,
#         split='test',
#         transform = val_augmentation
#     )
# )

# """
# training params
# """
# workers = 12
num_classes = 6 + 1
# # ignore_label = 255

# test_json_file='/home/gautam/e2e/lane_detection/2d_approaches/dataset/tusimple/test_label.json'

"""
2d model params
"""
featuremap_out_channel = 128
backbone = dict(
    type='ResNetWrapper',
    resnet_variant='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1],
    featuremap_out_channel = 128)

aggregator = dict(type= "SCNN")

heads = dict(type = 'PlainDecoder')


"""
3d model params (anchorless) for Apollo SIM3D dataset
"""
org_h = 1080
org_w = 1920
crop_y = 0
resize_h = 360 
resize_w = 480

ipm_h = 208 * 2
ipm_w = 128 * 2 
#init camera height and pitch
cam_height = 1.55
pitch = 3 #degrees

#camera instrinsics
K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
batch_norm = True #bev encoder
embedding_dim = 4 

##### Loss function params 
# Discriminative Loss 

#TODO: Change the values as per the GEOnet paper
delta_pull = 0.5 ## delta_v 
delta_push = 3.0 ## delta_d 
tile_size = 32

"""""
BIG BIG BIG NOTE: Is that the spatial size of ipm and BEV IPM projection are different?
Answer:: while cal gt the ipm_h and ipm_w are multiplied by a factor of 2. 
May be to process the masks for the section 3.2 I need not multiply the ipm_h and ipm_w by 2.
"""

# ###logging params
# date_it = "25_March_"
# train_run_name = "Anchorless3DLane" + date_it
# val_frequency = 400
# train_log_frequency = 200

# #Hyperparams
# epochs = 100
batch_size = 2
# l2_lambda = 1e-4
# log_frequency_steps = 200
# lr = 0.001 
# lrs_cd = 0
# lrs_factor = 0.75
# lrs_min = 1e-6
# lrs_patience = 3
# lrs_thresh = 1e-4
# prefetch_factor = 2
# bg_weight = 0.4 #used in the loss function to reduce the importance of one class in tusimple

def discriminative_loss(self, embedding, delta_c_gt):  
        
        """
        Embedding == f_ij 
        delta_c = del_i,j for classification if that tile is part of the lane or not
        """
        pull_loss = torch.tensor(0 ,dtype = embedding.dtype, device = embedding.device)
        push_loss = torch.tensor(0, dtype = embedding.dtype, device = embedding.device)
        
        #iterating over batches
        for b in range(embedding.shape[0]):
            
            embedding_b = embedding[b]  #---->(4,H*,W*)
            delta_c_gt_b = delta_c_gt[b] # will be a tensor of size (13,8) or whatever the grid size is consits of lane labels
            
            #delta_c_gt ---> [batch_size, 13, 8] where every element tells you which lane you belong too. 

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
                for r in range(0,embedding_b.shape[1],self.tile_size):
                    for c in range(0,embedding_b.shape[2],self.tile_size):
                        
                        f_ij = embedding_b[:,r:r+self.tile_size,c:c+self.tile_size] #----> (4,32,32) 
                        f_ij = f_ij.reshape(f_ij.shape[0], f_ij.shape[1]*f_ij.shape[2])
                        
                        #2. calculate mean for lane_c (mu_c) patchwise
                        mu_c = torch.sum(f_ij * delta_c[int(r/self.tile_size),int(c/self.tile_size)], dim = 1)/N_c #--> (4) mu for all the four embeddings
                        patchwise_mean.append(mu_c)
                        #3. calculate the pull loss patchwise
                        
                        pull_loss = pull_loss + torch.mean(F.relu( delta_c[int(r/self.tile_size),int(c/self.tile_size)] * torch.norm(f_ij-mu_c.reshape(4,1),dim = 0)- self.delta_pull)**2) / num_lanes
                        
                patchwise_centroid = torch.stack(patchwise_mean) #--> (32*32,4)
                patchwise_centroid = torch.mean(patchwise_centroid, dim =0) #--> (4)
                
                centroid_mean.append(patchwise_centroid)

            centroid_mean = torch.stack(centroid_mean) #--> (num_lanes,4)

            if num_lanes > 1:
                
                #4. calculate the push loss
                centroid_mean_A = centroid_mean.reshape(-1,1, 4)
                centroid_mean_B =centroid_mean.reshape(1,-1, 4)

                dist = torch.norm(centroid_mean_A-centroid_mean_B, dim = 2) #--> (num_lanes,num_lanes)
                dist = dist + torch.eye(num_lanes, dtype = dist.dtype, device = dist.device) * self.delta_push
                
                #divide by 2 to compensate the double loss calculation
                push_loss = push_loss + torch.sum(F.relu(-dist + self.delta_push)**2) / (num_lanes * (num_lanes-1)) / 2
        
        pull_loss= pull_loss / self.batch_size
        push_loss = push_loss / self.batch_size

        return pull_loss, push_loss