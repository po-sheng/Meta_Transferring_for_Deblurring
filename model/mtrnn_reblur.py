import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import conv1x1, conv3x3, conv5x5, upconv, actFunc, ResidualConvUnit 


class MTRNN_Reblur(nn.Module):
    def __init__(self, n_frames=3, n_layers=3, n_feats=32):
        super(MTRNN, self).__init__()
        
        self.n_frames = n_frames

        self.conv1_e = conv5x5(3*n_frames, n_feats)
        self.conv1 = nn.ModuleList([ResidualConvUnit(n_feats) for _ in range(n_layers)]) 

        self.conv2_e = conv5x5(n_feats, 2*n_feats, stride=2)
        self.conv2 = nn.ModuleList([ResidualConvUnit(2*n_feats) for _ in range(n_layers)])

        self.conv3_e = conv5x5(2*n_feats, 4*n_feats, stride=2)
        self.conv3 = nn.ModuleList([ResidualConvUnit(4*n_feats) for _ in range(n_layers)])
        
        self.conv4 = nn.ModuleList([ResidualConvUnit(4*n_feats) for _ in range(n_layers)])
    
        self.conv5_d = upconv(4*n_feats, 2*n_feats)
        self.conv5_c = conv5x5(4*n_feats, 2*n_feats)
        self.conv5 = nn.ModuleList([ResidualConvUnit(2*n_feats) for _ in range(n_layers)])
        
        self.conv6_d = upconv(2*n_feats, n_feats)
        self.conv6_c = conv5x5(2*n_feats, n_feats)
        self.conv6 = nn.ModuleList([ResidualConvUnit(n_feats) for _ in range(n_layers)])
        
        self.decode = conv3x3(n_feats, 3)

    def forward(self, x):
        _, _c, _, _ = x.size()
        c = _c // self.n_frames
        res = x[:, c * (self.n_frames//2):c * (self.n_frames//2 + 1), :, :]

        # Encoder
        # layer 1
        out1 = self.conv1_e(x)
        for layer in self.conv1:
            out1 = layer(out1)

        # layer 2
        out2 = self.conv2_e(out1)
        for layer in self.conv2:
            out2 = layer(out2)

        # layer 3
        out3 = self.conv3_e(out2)
        for layer in self.conv3:
            out3 = layer(out3)

        # layer 4
        out4 = out3
        for layer in self.conv4:
            out4 = layer(out4)
        
        # Decoder
        # layer 5
        out5 = self.conv5_c(torch.cat((self.conv5_d(out4), out2), dim=1))
        for layer in self.conv5:
            out5 = layer(out5)

        # layer 6
        out6 = self.conv6_c(torch.cat((self.conv6_d(out5), out1), dim=1))
        for layer in self.conv6:
            out6 = layer(out6)

        out = self.decode(out6)
        out = out + res

        return out
