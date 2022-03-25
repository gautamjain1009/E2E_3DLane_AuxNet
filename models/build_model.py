from models.registry import build_baseline
import torch 

def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)

def baseline(cfg): 
    model = build_baseline(cfg.net, cfg)

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
    
    return model
    
    
    

