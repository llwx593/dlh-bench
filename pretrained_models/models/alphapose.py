import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
import time
import os
from thop import profile
from ptflops import get_model_complexity_info

PretrainedURL = r"C:\Users\asus\.cache\torch\hub\checkpoints\fast_res50_256x192.pth"
__all__ = ["alphapose"]

class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes,
                 upscale_factor=2, norm_layer=nn.BatchNorm2d):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 reduction=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if reduction:
            self.se = SELayer(planes)
        self.reduc = reduction

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, reduction=False,
                 norm_layer=nn.BatchNorm2d,
                 dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            from .pe_utils import DeformConv, ModulatedDeformConv
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27

            self.conv2_offset = nn.Conv2d(
                planes,
                self.deformable_groups * offset_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                deformable_groups=self.deformable_groups,
                bias=False)

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class SEResnet(nn.Module):
    """ SEResnet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d,
                 dcn=None, stage_with_dcn=(False, False, False, False)):
        super(SEResnet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ["resnet18", "resnet50", "resnet101", 'resnet152']
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }
        self.inplanes = 64
        if architecture == "resnet18" or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_dcn = [dcn if with_dcn else None for with_dcn in stage_with_dcn]

        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion, momentum=0.1),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes,
                                stride, downsample, reduction=True,
                                norm_layer=self._norm_layer, dcn=dcn))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample,
                                norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=self._norm_layer, dcn=dcn))

        return nn.Sequential(*layers)


class FastPose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if 'CONV_DIM' in cfg.keys():
            self.conv_dim = cfg['CONV_DIM']
        else:
            self.conv_dim = 128
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if self.conv_dim == 256:
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

def get_cfg(new_path=None):
    base_path = r"pretrained_models/models/config/net.yaml"
    if new_path != None:
        base_path = new_path
    root_path = os.getcwd()
    file_path = os.path.join(root_path, base_path)
    with open(file_path) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def alphapose(cfg_file=None, pretrained=False, device_cfg="cpu"):
    cfg = get_cfg(cfg_file)
    model_cfg = cfg.MODEL
    preset_cfg = {
        "PRESET":cfg.DATA_PRESET
    }
    all_cfg = model_cfg.copy()
    _ = all_cfg.pop("TYPE")
    if preset_cfg is not None:
        for name, val in preset_cfg.items():
            all_cfg.setdefault(name, val)
    
    pose_model = FastPose(**all_cfg)
    if pretrained:
        pose_model.load_state_dict(torch.load(PretrainedURL, map_location=torch.device(device_cfg)))
    return pose_model

if __name__ == "__main__":
    model = alphapose()
    model.eval()
    img = torch.randn(1, 3, 256, 192)
    start_time = time.time()
    output_ans = model(img)
    end_time = time.time()
    duration = (end_time - start_time) * 1000
    print("duration is ", duration)
    macs, _ = profile(model, inputs=(img, ))
    print("thop ans is ", macs)
    macs, _ = get_model_complexity_info(model, (3,256,192), print_per_layer_stat=False)
    print("ptflop ans is ", macs)
    
