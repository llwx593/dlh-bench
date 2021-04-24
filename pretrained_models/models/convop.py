import torch
import torch.nn as nn

__all__ = ["cnsmall", "cnmid", "cnbig"]


# batch_size 64(224x224)
class ConvSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=2)
    
    def forward(self, x):
        return self.conv2d(x)

# batch_size 128(224x224)
class ConvMid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 128, kernel_size=3, stride=1)
    
    def forward(self, x):
        return self.conv2d(x)

# batch_size 256(224x224)
class ConvBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 256, kernel_size=3, stride=1)
    
    def forward(self, x):
        return self.conv2d(x)

def cnsmall():
    model = ConvSmall()
    return model

def cnmid():
    model = ConvMid()
    return model

def cnbig():
    model = ConvBig()
    return model
