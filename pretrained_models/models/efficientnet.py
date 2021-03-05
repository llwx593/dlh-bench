import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ['efficientnet_b3']

def efficientnet_b3():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    return model
