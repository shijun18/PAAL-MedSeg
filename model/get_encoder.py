import sys
sys.path.append('..')
import torch
from model.encoder import resnet_dropout,resnet_naive

moco_weight_path = {}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.endswith('_dropout'):
        backbone = resnet_dropout.__dict__[arch](classification=False,**kwargs)
    elif arch.endswith('_naive'):
        backbone = resnet_naive.__dict__[arch](classification=False,**kwargs)
    else:
        raise Exception('Architecture undefined!')

    if weights is not None and isinstance(moco_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(moco_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        if arch.startswith('resnet'):
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(">>>> loaded pre-trained model '{}' ".format(moco_weight_path[arch]))
        print(msg)
    
    return backbone