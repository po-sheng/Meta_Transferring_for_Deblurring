import torch
import torch.nn as nn

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=True)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=stride, bias=True)

def upconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 
            kernel_size=4, stride=2, padding=1, bias=True)
        
def actFunc(act="leaky_relu"):
    if act == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == "gelu":
        return nn.GELU()
    elif act == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise ValueError("You should specify a valid activation fuction")      

class ResidualConvUnit(nn.Module):
    def __init__(self, features, act="leaky_relu", use_bn=False):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = conv3x3(features, features)
        self.conv2 = conv3x3(features, features)
        self.use_bn = use_bn

        self.relu = actFunc(act)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.relu(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        out = out + res
        out = self.relu(out)
        
        return out
