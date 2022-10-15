import math
import copy
import torch
import torch.nn as nn

from .baseModel import BaseModel
from .basicModule import ReblurModel, DeblurModel


class Model(BaseModel):
    def __init__(self, args):
        self.args = args
        if self.args.meta:
#             self.deblur_lr = self.args.meta_lr
#             self.reblur_lr = self.args.meta_lr
            self.deblur_lr = self.args.deblur_lr
            self.reblur_lr = self.args.reblur_lr
            self.img_size = (self.args.support_size, self.args.support_size)
        else:
            self.deblur_lr = self.args.deblur_lr
            self.reblur_lr = self.args.reblur_lr
            self.img_size = (math.ceil(self.args.input_h/32)*32, math.ceil(self.args.input_w/32)*32)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0
        
        super(Model, self).__init__()
        self.video = args.video
        self.combine_update = args.combine_update               # 'cat', 'replace'
        
        self.min_bound = args.min_bound
        self.max_bound = args.max_bound

        self.deblur_model = DeblurModel(self.args.deblur_model)
        self.reblur_model = ReblurModel(self.args.reblur_model, self.args.n_frames, self.args.reblur_layers, self.args.features, img_size=self.img_size, use_attn=self.args.use_attn)

    def forward(self, blurs, sharps, warnup=False):
        # Concatenate adjacent image
        if self.args.reblur_model == "mtrnn":
            X = torch.cat(sharps, dim=1)
        elif self.args.reblur_model == "attn":
            X = torch.cat([sharp.unsqueeze(dim=1) for sharp in sharps], dim=1)       # B F C H W

        # Reblur
        ### pretrain meta_auxiliary
#         reblur = None
        reblur = self.reblur_model(X)
        latent = torch.clamp(reblur, self.min_bound, self.max_bound)

        deblur = None
        if not warnup:
            # Manual combine blur-sharp and reblur-sharp pair in batch
            """
            # reblur-sharp pair 
            if self.video:
                reblur_input = torch.stack((blurs), dim=1)
                reblur_input[:, len(blurs)//2, :, :, :] = torch.unsqueeze(latent, dim=1)
            else:
                reblur_input = latent

            # Deblur
            deblur = self.deblur_model(reblur_input)
            
            # blur-sharp pair
            if self.combine_update:
                if self.video:
                    blur_input = torch.cat([blur.unsqueeze(dim=1) for blur in blurs], dim=1)       # B F C H W
                else:
                    blur_input = blurs[len(blurs)//2]

                # Deblur
                deblur_batch2 = self.deblur_model(blur_input)
            
                # combine two results
                deblur_list = []
                for a, b in zip(deblur, deblur_batch2):
                    deblur_list += [a, b]
                deblur = torch.cat(deblur_list, dim=0)

            return reblur, deblur
            """

            # Combine blur-sharp and reblur-sharp pair into torch batch
            if self.video:
                if self.combine_update:
                    blur_tensor = torch.stack((blurs), dim=1)
                    
                    mix = copy.deepcopy(blur_tensor)
                    mix[:, len(blurs)//2, :, :, :] = latent
                    spt_mix = torch.split(mix, 1)

                    spt_blur = torch.split(blur_tensor, 1)

                    mix_list = []
                    for a, b in zip(spt_mix, spt_blur):
                        mix_list += [a, b]
                    mix = torch.cat(mix_list, dim=0)

                else: 
                    mix = torch.stack((blurs), dim=1)
                    mix[:, len(blurs)//2, :, :, :] = torch.unsqueeze(latent, dim=1)
            else:
                if self.combine_update:
                    # Concatenate original blur and fake blur as deblur input
                    spt_blur = torch.split(blurs[len(blurs) // 2], 1)
                    spt_latent = torch.split(latent, 1)
                
                    mix_list = []
                    for a, b in zip(spt_latent, spt_blur):
                        mix_list += [a, b]
                    mix = torch.cat(mix_list, dim=0)
                else:
                    ### pretrain meta-auxiliary
                    mix = latent
#                     mix = blurs[len(blurs) // 2]

            # Deblur
            deblur = self.deblur_model(mix)

        return reblur, deblur

    
def get_model(args):
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = Model(args).to(device)

    # load model weight
    start_epoch = model.load_model(args.deblur_model_path, args.reblur_model_path)
    
    # parallel
    if ((not args.cpu) and (args.gpu_num > 1)):
        model = nn.DataParallel(model)
    
    return model, start_epoch
