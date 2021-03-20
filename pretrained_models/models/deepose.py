import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from thop import profile
from ptflops import get_model_complexity_info
import time

PretrainedURL = r"~\.cache\torch\hub\checkpoints\model"
__all__ = ["deeppose"]

class DeepPose(nn.Module):
	"""docstring for DeepPose"""
	def __init__(self, nJoints, modelName='resnet50'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x)


def load_pretrained_model(model, weight_path):
    if weight_path != None:
        _ = model.load_state_dict(weight_path)
        return
    state_dict = model_zoo.load_url(PretrainedURL, map_location=torch.device("cpu"))
    _ = model.load_state_dict(state_dict)

def deeppose(pretrained = False, weight_path = None, nJoints = 7):
	model = DeepPose(nJoints)
	if pretrained:
		model = load_pretrained_model(model, weight_path)
	return model

if __name__ == "__main__":
	model = deeppose()
	model.eval()
	img = torch.randn(1, 3, 224, 224)
	macs, _ = profile(model, inputs=(img, ))
	print("mac is ", macs)
	macs, _ = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False)
	print("mac is ", macs)
	start_time = time.time()
	_ = model(img)
	end_time = time.time()
	print((end_time - start_time) * 1000)