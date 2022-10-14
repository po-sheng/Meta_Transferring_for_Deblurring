import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func

from .arch import conv1x1, conv3x3, conv5x5, upconv, actFunc, ResidualConvUnit 
from .region_vit import RegionViT
from thop import profile

class VideoAdapt(nn.Module):
    def __init__(self, n_frames, n_layers, n_feats, img_size=(64, 64), use_attn=True):
        super(VideoAdapt, self).__init__()
        self.use_attn = use_attn

        self.n_frames = n_frames

        self.conv1_e = conv5x5(3, n_feats)
        self.conv1 = nn.ModuleList([ResidualConvUnit(n_feats) for _ in range(n_layers)]) 
#         self.attn1 = RegionViT(img_size=(img_size[0], img_size[1]), num_frames=n_frames, in_chans=n_feats, embed_dim=n_feats*2, patch_size=4, window_size=8)

        self.conv2_e = conv5x5(n_feats, 2*n_feats, stride=2)
        self.conv2 = nn.ModuleList([ResidualConvUnit(2*n_feats) for _ in range(n_layers)])
#         self.attn2 = RegionViT(img_size=(img_size[0]//2, img_size[1]//2), num_frames=n_frames, in_chans=n_feats*2, embed_dim=n_feats*4, patch_size=2, window_size=8)

        self.conv3_e = conv5x5(2*n_feats, 4*n_feats, stride=2)
        self.conv3 = nn.ModuleList([ResidualConvUnit(4*n_feats) for _ in range(n_layers)])
        # attn
        self.attn3 = RegionViT(img_size=(img_size[0]//4, img_size[1]//4), num_frames=n_frames, in_chans=n_feats*4, embed_dim=n_feats*8, depths=[4])
        # 3d conv
        if not self.use_attn:
            self.threed_embed = conv3x3(n_feats*4, n_feats*8)
            self.threed_conv = nn.ModuleList([nn.Conv3d(n_feats*8, n_feats*8, kernel_size=(5, 3, 3), padding=(0, 1, 1), padding_mode="reflect") for i in range(3)])
            self.threed_decode = conv3x3(n_feats*8, n_feats*4)

        self.conv4 = nn.ModuleList([ResidualConvUnit(4*n_feats) for _ in range(n_layers)])

        self.conv5_d = upconv(4*n_feats, 2*n_feats)
        self.conv5_c = conv5x5(4*n_feats, 2*n_feats)
        self.conv5 = nn.ModuleList([ResidualConvUnit(2*n_feats) for _ in range(n_layers)])
        
        self.conv6_d = upconv(2*n_feats, n_feats)
        self.conv6_c = conv5x5(2*n_feats, n_feats)
        self.conv6 = nn.ModuleList([ResidualConvUnit(n_feats) for _ in range(n_layers)])
        
        self.decode = conv3x3(n_feats, 3)

    def forward(self, x):
        B, F, C, H, W = x.size()
        res = x[:, F//2, :, :, :]
        x = x.contiguous().view(-1, C, H, W)
        
        # Encoder
        # layer 1
        out = self.conv1_e(x)
        for layer in self.conv1:
            out = layer(out)

        # Atten block 1
#         if self.use_attn:
#             out = self.attn1(out)
        out1_res = out.reshape(B, F, out.size()[1], out.size()[2], out.size()[3])[:, F//2, :, :, :]

        # layer 2
        out = self.conv2_e(out)
        for layer in self.conv2:
            out = layer(out)

        # Atten block 2
#         if self.use_attn:
#             out = self.attn2(out)
        out2_res = out.reshape(B, F, out.size()[1], out.size()[2], out.size()[3])[:, F//2, :, :, :]

        # layer 3
        out = self.conv3_e(out)
        for layer in self.conv3:
            out = layer(out)

        # Atten block 3
        if self.use_attn:
            _, c, h, w = out.size()
            out = self.attn3(out)
            out = out.contiguous().view(B, F, c, h, w)[:, F//2, :, :, :]
        else:
            out = self.threed_embed(out)
            _, c, h, w = out.size()
            out = out.reshape(B, F, c, h, w).permute(0, 2, 1, 3, 4)
            full = out
            for m in self.threed_conv:
                out = m(full)
                out = Func.pad(out, (0, 0, 0, 0, self.n_frames//2, self.n_frames//2))
                full = full + out
            out = self.threed_decode(out[:, :, F//2, :, :])

        # Decoder
        # layer 4
        for layer in self.conv4:
            out = layer(out)
        
        # layer 5
        out = self.conv5_c(torch.cat((self.conv5_d(out), out2_res), dim=1))
        for layer in self.conv5:
            out = layer(out)

        # layer 5
        out = self.conv6_c(torch.cat((self.conv6_d(out), out1_res), dim=1))
        for layer in self.conv6:
            out = layer(out)

        out = self.decode(out)
        out = out + res

        return out

if __name__ == '__main__':
    # Debug
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = VideoAdapt(5, 3, 32, (256, 256)).cuda()
    # summary(net, (3, 256, 256))
    in_data = torch.randn(1, 5, 3, 256, 256).cuda()
    flops, params = profile(net, (in_data, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # stat(net, (3, 256, 256))
    # macs, params = get_model_complexity_info(net.cuda(), (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #start = time.time()
#     net(torch.randn(1, 3, 256, 256))
    #stop = time.time()
    #print(stop - start)
