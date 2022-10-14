from utils import calc_psnr_and_ssim

import os
import copy
import tqdm
import time
import math
import random
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all, optim, lr_scheduler):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.model_obj = model.module if ((not self.args.cpu) and (self.args.gpu_num > 1)) else model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')

        self.optimizer = optim
        self.lr_scheduler = lr_scheduler
    
    # Prepare data on device
    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            for i, frame in enumerate(sample_batched[key]):
                sample_batched[key][i] = frame.to(self.device)
        return sample_batched

    # Training scheme
    def train(self, current_epoch=0):
        self.model.train()

        # Freeze deblur model weight first if alternative training is true
        if self.args.alt_update:
            for param in self.model_obj.deblur_model.parameters():
                param.requires_grad = True
            for param in self.model_obj.reblur_model.parameters():
                param.requires_grad = False

        # Param init
        running_reblur_loss = 0.0
        running_cycle_loss = 0.0
        running_gan_loss = 0.0
        self.model_obj.lr = self.optimizer.param_groups[0]['lr']
       
        # Set tq for iteration
        tq = tqdm.tqdm(self.dataloader['train'], total=len(self.dataloader['train']))
        tq.set_description('Epoch {}, lr {:.4}'.format(current_epoch, self.model_obj.lr))
        for i, data_tup in enumerate(tq):
            data, blurrers = data_tup
            blurrers = blurrers['blurrers']

            self.optimizer.zero_grad()
            
            # Alternative freeze part model weight if alternative training is true
            if self.args.alt_update:
                for param in self.model_obj.parameters():
                    param.requires_grad ^= True
            
            # Fetch data
            data = self.prepare(data)
            blurs, sharps= data['blurs'], data['sharps']
            
            # data run through model
            loss = 0
            latent, output = self.model(blurs, sharps, warnup=(current_epoch<=self.args.deblur_warnup))
            ### pretrain meta_auxiliary
            reblur_loss = self.loss_all['reblur_loss'](latent, blurs[len(blurs) // 2])
           
            # Adding different loss function with diff model after deblur warnup
            if current_epoch > self.args.deblur_warnup:
                if self.args.video:
                    if self.args.combine_update:
                        targets = []
                        for j, sharp in enumerate(sharps):
                            targets.append(torch.repeat_interleave(sharp, 2, dim=0))
                        cycle_loss = self.loss_all['deblur_loss'](output, targets, blurs)
                    else:
                        cycle_loss = self.loss_all['deblur_loss'](output, sharps, blurs)
                else:
                    if self.args.combine_update:
                        target = torch.repeat_interleave(sharps[len(sharps) // 2], 2, dim=0)
                        cycle_loss = self.loss_all['deblur_loss'](output, target, blurs[len(blurs) // 2])
                    else:
                        cycle_loss = self.loss_all['deblur_loss'](output, sharps[len(sharps) // 2], blurs[len(blurs) // 2])

            # Adding gan loss to reblur model
            if self.args.gan:
                if self.args.n_frames != 1:
                    rand = random.randint(0, len(blurs)-1)
                    while(rand == len(blurs) // 2):
                        rand = random.randint(0, len(blurs)-1)
                    gan_loss = self.loss_all['gan_loss'](i, latent, blurs[rand])
                else:
                    gan_loss = self.loss_all['gan_loss'](i, latent, blurrers[0])

            # Calc loss
            ### pretrain meta_auxiliary
            loss = (self.args.reblur_ratio * reblur_loss)
            if current_epoch > self.args.deblur_warnup:
                loss += (1-self.args.reblur_ratio) * cycle_loss
            if self.args.gan:
                loss += self.args.gan_ratio * gan_loss
                        
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            if current_epoch > self.args.deblur_warnup:
                running_cycle_loss += cycle_loss.item()
            ### pretrain meta_auxiliary
            running_reblur_loss += reblur_loss.item()
            
            if self.args.gan:
                running_gan_loss += gan_loss.item()
                tq.set_postfix(cycle=(running_cycle_loss/(i+1)), reblur=(running_reblur_loss/(i+1)), GAN=(running_gan_loss/(i+1)))
            else:
                tq.set_postfix(cycle=(running_cycle_loss/(i+1)), reblur=(running_reblur_loss/(i+1)))

        if self.args.scheduler == "plateau":
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

        # Save model
        if current_epoch % self.args.save_every == 0:
            self.model_obj.save_model(current_epoch, skip_deblur=(current_epoch <= self.args.deblur_warnup))
            
            if self.args.gan:
                # save discriminator for GAN
                tmp = self.loss_all['gan_loss'].gan.vgg19.state_dict()
                model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp}
                model_name = self.args.save_dir.strip('/')+'/model/gan_'+str(current_epoch).zfill(5)+'.pt'
                torch.save(model_state_dict, model_name)

    # Evaluate // test scheme
    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        
        # Create whole img map if not inference at full img size
        if not self.args.full_img_exp:
            self.W = int(math.ceil(self.args.img_w / self.args.input_w))
            self.H = int(math.ceil(self.args.img_h / self.args.input_h))

            pad_img = torch.zeros(1, 3, self.args.input_h*self.H, self.args.input_w*self.W)
            pad_gt = torch.zeros(1, 3, self.args.input_h*self.H, self.args.input_w*self.W)
            if self.args.reblur_result:
                pad_reblur = torch.zeros(1, 3, self.args.input_h*self.H, self.args.input_w*self.W)

        # start inference
        if self.args.dataset == "GOPRO" or self.args.dataset == "DVD" or self.args.dataset == "REDS" or self.args.dataset == "RealBlur":
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, lpips, r_psnr, r_ssim, r_lpips, cnt = 0., 0., 0., 0., 0., 0., 0
                
                # Prepare tq for iteration
                tq = tqdm.tqdm(self.dataloader['test'], total=len(self.dataloader['test']))
                tq.set_description('Evaluating...')
                for i, data_tup in enumerate(tq):
                    
                    # Get data and calc patch position
                    if not self.args.full_img_exp:
                        pch_idx = int(i % (self.W * self.H))
                        row = int(pch_idx // self.W)
                        col = int(pch_idx - (row * self.W))
                    
                    if self.args.dataset == "RealBlur": 
                        data, name, h, w = data_tup
                    else:
                        data, name = data_tup
                    data = self.prepare(data)
                    blurs, sharps = data['blurs'], data['sharps']

                    # Save reblur result
                    if self.args.reblur_result and self.args.dataset != "RealBlur":
                        if self.args.reblur_model == "mtrnn":
                            cat_blur = torch.cat(blurs, dim=1)
                        elif self.args.reblur_model == "attn":
                            cat_blur = torch.cat([blur.unsqueeze(dim=1) for blur in blurs], dim=1)       # B F C H W
                        ### pretrain meta_auxiliary
                        reblur = self.model_obj.reblur_model(cat_blur)
                        
                        # check data normalization range
                        ### pretrain meta_auxiliary
                        reblur = torch.clamp(reblur, self.args.min_bound, self.args.max_bound)

                    # Get deblurred result
                    if self.args.video:
                        output = self.model_obj.deblur_model(torch.stack(blurs, dim=1))
                    else:
                        output = self.model_obj.deblur_model(blurs[len(blurs) // 2])
                    
                    # check data normalization range
                    if self.args.deblur_model == "mprnet":
                        output = torch.clamp(output[0], self.args.min_bound, self.args.max_bound)
                    elif self.args.deblur_model == "mimo" or self.args.deblur_model == "mimoPlus":
                        output = torch.clamp(output[2], self.args.min_bound, self.args.max_bound)
                    elif self.args.deblur_model == "cdvd_tsp":
                        output = torch.clamp(output[3], self.args.min_bound, self.args.max_bound)
                    elif self.args.deblur_model == "meta":
                        output = torch.clamp(output[-1][0], self.args.min_bound, self.args.max_bound)
                    else:
                        output = torch.clamp(output, self.args.min_bound, self.args.max_bound)

                    # Get sharp GT
                    sharp = sharps[len(sharps)//2]
                    
                    # Form the full img if not inference at full img size
                    if not self.args.full_img_exp:
                        pad_img[:, :, row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w] = output[:, :, :self.args.input_h, :self.args.input_w]
                        pad_gt[:, :, row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w] = sharp[:, :, :self.args.input_h, :self.args.input_w]
                        if self.args.reblur_result:
                            pad_reblur[:, :, row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w] = reblur[:, :, :self.args.input_h, :self.args.input_w]
                    
                    ### calculate psnr and ssim
                    if self.args.full_img_exp or pch_idx == self.W*self.H - 1:
                        if self.args.full_img_exp:
                            full_img = output
                            full_gt = sharp
                            if self.args.reblur_result and self.args.dataset != "RealBlur":
                                full_reblur = reblur
                        else:
                            full_img = pad_img[:, :, 0:self.args.img_h, 0:self.args.img_w]
                            full_gt = pad_gt[:, :, 0:self.args.img_h, 0:self.args.img_w]
                            if self.args.reblur_result:
                                full_reblur = pad_reblur[:, :, 0:self.args.img_h, 0:self.args.img_w]
                        
                        if self.args.dataset == "RealBlur":
                            full_img = full_img[:, :, :h, :w]
                            full_gt = full_gt[:, :, :h, :w]

                        # Save img
                        if (self.args.save_results):
                            img_save = full_img
                            if self.args.centralized:
                                img_save = img_save + self.args.max_bound
                            if self.args.normalized:
                                img_save = (img_save / (self.args.max_bound - self.args.min_bound)) * 255

                            img_save = np.transpose(img_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                            imsave(os.path.join(self.args.save_dir, 'save_results', name[0]+'.png'), img_save)
                            if self.args.reblur_result:
                                reblur_save = full_reblur
                                if self.args.centralized:
                                    reblur_save = reblur_save + self.args.max_bound
                                if self.args.normalized:
                                    reblur_save = (reblur_save / (self.args.max_bound - self.args.min_bound)) * 255
                        
                                reblur_save = np.transpose(reblur_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                                imsave(os.path.join(self.args.save_dir, 'save_results', name[0]+'_reblur.png'), reblur_save)
                
                        # Calc inference result PSNR/SSIM
                        cnt += 1
                        _psnr, _ssim, _lpips = calc_psnr_and_ssim(self.args, full_img.detach(), full_gt.detach())
                        if self.args.reblur_result and self.args.dataset != "RealBlur":
                            reblur_psnr, reblur_ssim, reblur_lpips = calc_psnr_and_ssim(self.args, sharps[len(sharps) // 2].detach(), full_reblur.detach())
                       
                        psnr += _psnr
                        ssim += _ssim
                        lpips += _lpips
                        if self.args.reblur_result:
                            r_psnr += reblur_psnr
                            r_ssim += reblur_ssim
                            r_lpips += reblur_lpips

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                lpips_ave = lpips / cnt
                if self.args.reblur_result and self.args.dataset != "RealBlur":
                    r_psnr_ave = r_psnr / cnt
                    r_ssim_ave = r_ssim / cnt
                    r_lpips_ave = r_lpips / cnt
                
                # Record max PSNR and SSIM
                self.logger.info('Ref  PSNR (now): %.3f, \t SSIM (now): %.4f, \t LPIPS (now): %.4f' %(psnr_ave, ssim_ave, lpips_ave))
                if self.args.reblur_result:
                    self.logger.info('Reblur  PSNR (now): %.3f, \t SSIM (now): %.4f, \t LPIPS (now): %.4f' %(r_psnr_ave, r_ssim_ave, r_lpips_ave))
                if (psnr_ave > self.model_obj.max_psnr):
                    self.model_obj.max_psnr = psnr_ave
                    self.model_obj.max_psnr_epoch = current_epoch
                if (ssim_ave > self.model_obj.max_ssim):
                    self.model_obj.max_ssim = ssim_ave
                    self.model_obj.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.model_obj.max_psnr, self.model_obj.max_psnr_epoch, self.model_obj.max_ssim, self.model_obj.max_ssim_epoch))

        self.logger.info('Evaluation over.')

