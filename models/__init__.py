from .baseline_model import ERFNet
from .backbones import resnet, senet
from .feature_aggregataor import scnn_module, resa
from .heads import plain_decoder, busd

from .registry import build_baseline
from .registry import build_backbone
from .registry import build_heads
from .registry import build_aggregator
