import torch
import torch.nn as nn

__all__ = ["matmul256", "matmul1024", "matmul4096"]

class MatMulNet256(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = nn.Linear(256, 256)

    def forward(self, x):
        return self.matmul(x)


class MatMulNet1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.matmul(x)


class MatMulNet4096(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.matmul(x)

def matmul256():
    model = MatMulNet256()
    return model

def matmul1024():
    model = MatMulNet1024()
    return model

def matmul4096():
    model = MatMulNet4096()
    return model
