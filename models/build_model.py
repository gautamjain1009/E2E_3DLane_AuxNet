from models.registry import build_baseline, build_aggregator, build_heads, build_backbone

import torch 
import torch.nn as nn

def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)

class CombinedModel(nn.Module):
    def __init__(self, cfg):
        super(CombinedModel,self).__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.aggregator = build_aggregator(cfg)
        self.heads = build_heads(cfg)

    def forward(self, x):
        x = self.backbone(x) # list of tensors
        x = self.aggregator(x[-1])
        x = self.heads(x)
        return x

def load_model(cfg, baseline = False): 
    
    if baseline == True:
        model = build_baseline(cfg.net, cfg)

        #reinitialize model weights
        for name, layer in model.named_modules():
            if isinstance(layer,torch.nn.Conv2d):
                reinitialize_weights(layer.weight)
                try: 
                    layer.bias.data.fill_(0.01)  
                except: ## TODO: Verify one layers bias is NONE for baseline 
                    pass 
                
            elif isinstance(layer, torch.nn.Linear):
                reinitialize_weights(layer.weight)
                layer.bias.data.fill_(0.01)
        return model
 
    else: 
        # build combined model with backbone, aggegator, and heads
        model = CombinedModel(cfg)        
        
        return model