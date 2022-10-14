import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import conv1x1, conv3x3, conv5x5, upconv, actFunc, ResidualConvUnit 
from .mtrnn_reblur import MTRNN_Reblur
from .MTRNN import MTRNN
from .MPRNet import MPRNet
from .MIMOUNet import MIMOUNet, MIMOUNetPlus
from .Restormer import Restormer 
from .VideoAdapt import VideoAdapt
# from .CDVD_TSP.cdvd_tsp import CDVD_TSP

class ReblurModel(nn.Module):
    def __init__(self, reblur_model, n_frames, n_layers, n_feats, img_size=(64, 64), use_attn=True):
        super(ReblurModel, self).__init__()
        
        if reblur_model == 'mtrnn':
            self.model = MTRNN_Reblur(n_frames, n_layers, n_feats)
        elif reblur_model == 'attn':
            self.model = VideoAdapt(n_frames, n_layers, n_feats, img_size=img_size, use_attn=use_attn) 
        else:
            raise ValueError("Reblur model {} not recognized.".format(reblur_model))

    def forward(self, x):
        
        out = self.model(x)

        return out

class DeblurModel(nn.Module):
    def __init__(self, deblur_model):
        super(DeblurModel, self).__init__()
    
        self.deblur_model = deblur_model

        if deblur_model == 'mprnet':
            self.model = MPRNet()
        elif deblur_model == 'mimo':
            self.model = MIMOUNet()
        elif deblur_model == 'mimoPlus':
            self.model = MIMOUNetPlus()
        elif deblur_model == 'restormer':
            self.model = Restormer()
        elif deblur_model == 'mtrnn':
            self.model = MTRNN()
#         elif deblur_model == 'cdvd_tsp':
#             self.model = CDVD_TSP()
        else:
            raise ValueError("Deblur model {} not recognized.".format(self.deblur_model))

    def forward(self, x):
        if self.deblur_model == 'mtrnn':
            # data range 0_1 shift to 0_255
            x = x * 255

            iter_num = 6
            feature_1 = x.clone()
            feature_2 = x.clone()
            lr_d = x.clone()

            for itera in range(iter_num):
                if itera != 0:
                    lr_d = lr_d.data
                    feature_1 = feature_1.data
                    feature_2 = feature_2.data
                output = self.model([x, lr_d, feature_1, feature_2])
                lr_d, feature_1, feature_2 = output
            out = lr_d

            # data range 0_255 shift to 0_1
            out = out / 255
        else:
            out = self.model(x)

        return out
