from models.registry import build_baseline, build_aggregator, build_heads, build_backbone

import torch 
import torch.nn as nn


class Combined2DModel(nn.Module):
    def __init__(self, cfg):
        super(Combined2DModel,self).__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.aggregator = build_aggregator(cfg)
        self.heads = build_heads(cfg)

    def forward(self, x):
        x = self.backbone(x) # list of tensors0
        # print("check the shape of the last feature map: ", x[-1].shape)
        x = self.aggregator(x[-1])
        x = self.heads(x)
        return x

def load_model(cfg, baseline = False, pretrained = False):     
    
    #TODO: as per the pretrained param reinitialize the weights of the every model 
    if baseline == True:
        model = build_baseline(cfg.net, cfg)

    else: 
        # build combined model with backbone, aggegator, and heads
        model = Combined2DModel(cfg)        
    
    if pretrained == False:
        model.apply(initialize_weights)
        print("=> Initialized 2d model weights")
    else:
        # load the pretrained weights 
        print("=>loaded the pretrained weights for 2d model")
        model.load_state_dict(torch.load(cfg.pretrained_2dmodel_path))
     
    return model

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
    
    