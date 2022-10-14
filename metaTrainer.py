from utils import calc_psnr_and_ssim

import os
import copy
import tqdm
import time
import math
import random
import shutil
import numpy as np
from imageio import imread, imsave

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, cur_model_sd, loss_all, optim, lr_scheduler, cur_video=None, video_idx=0):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.model_obj = model.module if ((not self.args.cpu) and (self.args.gpu_num > 1)) else model
        self.meta_model_sd = copy.deepcopy(self.model.state_dict())
        self.cur_model_sd = cur_model_sd
        self.loss_all = loss_all
        self.cur_video = cur_video
        self.video_idx = video_idx
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.optimizer = optim
        self.lr_scheduler = lr_scheduler

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            for i, frame in enumerate(sample_batched[key]):
                sample_batched[key][i] = frame.to(self.device)
        return sample_batched

    def support(self, current_epoch=0):
        # set inner tenporal model
        inner_model = copy.deepcopy(self.model)
        inner_model_obj = inner_model.module if ((not self.args.cpu) and (self.args.gpu_num > 1)) else inner_model
        inner_model.train()
        
        # set inner optimizer 
        if ((not self.args.cpu) and (self.args.gpu_num > 1)):
            r_params = inner_model.module.reblur_model.parameters()
            d_params = inner_model.module.deblur_model.parameters()
        else:
            r_params = inner_model.reblur_model.parameters()
            d_params = inner_model.deblur_model.parameters()
        
        # Set reblur model require grad false
        if (not self.args.gan) and (not self.args.cycle_update) and (not self.args.reblur_backward):
            for param in inner_model_obj.reblur_model.parameters():
                param.requires_grad = False

        # Different optimizer
        if self.args.support_optim == 'adam':
            if self.args.use_inner_lr:
                inner_optim = optim.Adam([{'params': r_params, 'lr': self.args.reblur_lr},
                                {'params': d_params}], lr=self.args.inner_lr)
            else:
                inner_optim = optim.Adam([{'params': r_params, 'lr': self.args.reblur_lr},
                                {'params': d_params}], lr=self.args.deblur_lr)
#             inner_optim = optim.Adam(params, lr=self.args.inner_lr)
        elif self.args.support_optim == 'sgd':
            inner_optim = optim.SGD(params, lr=self.args.inner_lr)
        elif self.args.support_optim == 'adamW':
            inner_optim = optim.AdamW(params, lr=self.args.inner_lr)
        elif self.args.support_optim == 'RMSprop':
            inner_optim = optim.RMSprop(params, lr=self.args.inner_lr)
        elif self.args.support_optim == 'adagrad':
            inner_optim = optim.Adagrad(params, lr=self.args.inner_lr)
        elif self.args.support_optim == 'adadelta':
            inner_optim = optim.Adadelta(params, lr=self.args.inner_lr)
        
        inner_optim.zero_grad()
        psnr_before, psnr_deblur, psnr_finetune = 0., 0., 0.
        psnr, ssim, lpips = 0.0, 0.0, 0.0
        data_save = []
        
        # pre save support patch
        for i, data in enumerate(self.dataloader['train']):
            data_save.append(data)
           
        # reblur model meta support set
        if self.args.gan or self.args.cycle_update:
            if self.args.cycle_block:
                for param in inner_model_obj.deblur_model.parameters():
                    param.requires_grad = False
            
            for _ in range(self.args.reblur_epochs):
                running_loss = 0
                running_gan_loss = 0
                random.shuffle(data_save)
            
                tq = tqdm.tqdm(copy.deepcopy(data_save), total=len(data_save), disable=(self.args.n_frames==1))
                if self.args.use_inner_lr:
                    tq.set_description('Reblur support {}, Video {}, lr {}'.format(current_epoch, self.cur_video, self.args.inner_lr))
                else:
                    tq.set_description('Reblur support {}, Video {}, lr {}'.format(current_epoch, self.cur_video, self.model.reblur_lr))
                for i, save_data in enumerate(tq):
                    if i % self.args.support_batch == 0:
                        inner_optim.zero_grad()
                    
                    data = self.prepare(save_data)
                    blurs, sharps, blurrers = data['blurs'], data['sharps'], data['blurrers']
                        
                    loss = 0
                    inner_optim.zero_grad()
                    
                    #*************************************************************** blurs -> sharps
                    # create reblur data
                    if self.args.reblur_model == "mtrnn":
                        cat_blur = torch.cat(blurs, dim=1)
                    elif self.args.reblur_model == "attn":
                        cat_blur = torch.cat([blur.unsqueeze(dim=1) for blur in blurs], dim=1)       # B F C H W    
                    reblur = inner_model_obj.reblur_model(cat_blur)

                    # gan udpate
                    if self.args.gan: 
                        if self.args.use_blurrest:
                            gan_loss = self.loss_all['gan_loss'](i, reblur, blurrers[0])
                        else:
                            rand = random.randint(0, len(blurs)-1)
                            while(rand == len(blurs) // 2):
                                rand = random.randint(0, len(blurs)-1)
                            gan_loss = self.loss_all['gan_loss'](i, reblur, blurs[rand])
                        loss += self.args.gan_ratio * gan_loss
    
                    # cycle update
                    if self.args.cycle_update:
                        reblur = reblur.clamp(self.args.min_bound, self.args.max_bound)
                        
#                         if self.args.reblur_backward or self.args.cycle_block:
                        if self.args.video:
                            mix = torch.stack((blurs), dim=1)
                            mix[:, len(blurs)//2, :, :, :] = reblur
                            deblur = inner_model_obj.deblur_model(mix)
                        else:
                            deblur = inner_model_obj.deblur_model(reblur)
#                         else:
#                             deblur = inner_model_obj.deblur_model(reblur.detach())
#                         deblur = inner_model_obj.deblur_model(reblur)
                        cycle_loss = self.loss_all['deblur_loss'](deblur, blurs[len(blurs) // 2], reblur) 

                        loss += (1 - self.args.gan_ratio) * cycle_loss
    
                    loss /= self.args.support_batch
                    loss.backward()
                    if (i+1) % self.args.support_batch == 0:
                        inner_optim.step()
                    
                    running_loss += loss.item()
                    if self.args.gan:
                        running_gan_loss += gan_loss.item()
                    tq.set_postfix(gan_loss=(running_gan_loss/(i+1)), loss=(running_loss/(i+1)))
        
        if self.args.cycle_block:
            for param in inner_model_obj.deblur_model.parameters():
                param.requires_grad = True
        
        if self.args.use_inner_lr:
            if ((not self.args.cpu) and (self.args.gpu_num > 1)):
                r_params = inner_model.module.reblur_model.parameters()
                d_params = inner_model.module.deblur_model.parameters()
            else:
                r_params = inner_model.reblur_model.parameters()
                d_params = inner_model.deblur_model.parameters()

            inner_opt_sd = copy.deepcopy(inner_optim.state_dict())
            inner_optim = optim.Adam([{'params': r_params, 'lr': self.args.reblur_lr},
                        {'params': d_params}], lr=self.args.deblur_lr)
            inner_optim.load_state_dict(inner_opt_sd)
            inner_optim.zero_grad()
        
        # deblur model meta support set
        for n_epochs in range(self.args.support_epochs):
            running_loss = 0
            random.shuffle(data_save)
            
            tq = tqdm.tqdm(copy.deepcopy(data_save), total=len(data_save), disable=(self.args.n_frames==1))
            if self.args.use_inner_lr:
                tq.set_description('Deblur support {}, Video {}, lr {}'.format(current_epoch, self.cur_video, self.args.inner_lr))
            else:
                tq.set_description('Deblur support {}, Video {}, lr {}'.format(current_epoch, self.cur_video, self.model.deblur_lr))
            for i, save_data in enumerate(tq):
                if i % self.args.support_batch == 0:
                    inner_optim.zero_grad()
                
                data = self.prepare(save_data)
                blurs, sharps, _ = data['blurs'], data['sharps'], data['blurrers']
                
                # Use reblur-blur pair or blur-sharp pair as support set
                loss = 0
                if self.args.meta_train and not self.args.use_reblur_pair:
                    output = inner_model_obj.deblur_model(blurs[len(blurs) // 2])
                    loss = self.loss_all['deblur_loss'](output, sharps[len(sharps) // 2], blurs[len(blurs) // 2])
                
                else:
                    #********************************************************** blurs -> sharps
                    # Get reblur img of the original blur
                    if self.args.reblur_model == "mtrnn":
                        cat_blur = torch.cat(blurs, dim=1)
                    elif self.args.reblur_model == "attn":
                        cat_blur = torch.cat([blur.unsqueeze(dim=1) for blur in blurs], dim=1)       # B F C H W    
                    reblur = inner_model_obj.reblur_model(cat_blur)
                    
                    # Clamp reblur to data range
                    reblur = reblur.clamp(self.args.min_bound, self.args.max_bound)

                    # Evaluate and save reblur img
                    if n_epochs == 0 and self.args.reblur_result:
                        _psnr, _ssim, _lpips = calc_psnr_and_ssim(self.args, blurs[len(blurs) // 2].detach(), sharps[len(sharps) // 2].detach())
                        psnr += _psnr
                        ssim += _ssim
                        lpips += _lpips
                        
                        # Save reblur result
                        reb_save = copy.deepcopy(reblur.detach())
                        img_save = copy.deepcopy(blurs[len(blurs)//2])
                        if self.args.centralized:
                            reb_save = reb_save + self.args.max_bound
                            img_save = img_save + self.args.max_bound
                        if self.args.normalized:
                            reb_save = (reb_save / (self.args.max_bound - self.args.min_bound)) * 255
                            img_save = (img_save / (self.args.max_bound - self.args.min_bound)) * 255
                        
                        img_save = np.transpose(img_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', self.args.cur_video+"_"+str(i)+'.png'), img_save)
                        reb_save = np.transpose(reb_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', self.args.cur_video+"_"+str(i)+'_reblur.png'), reb_save)
                    
                    # reblur patch will gradient back when meta training
                    if self.args.reblur_backward:
                        output = inner_model_obj.deblur_model(reblur)
                    else:
                        if self.args.round:
                            reblur = ((reblur + self.args.max_bound) / (self.args.max_bound - self.args.min_bound) * 255).type(torch.uint8)
                            reblur = reblur.type(torch.float)
                            if self.args.centralized:
                                reblur = reblur - 127.5
                            if self.args.normalized:
                                reblur = reblur / 255.
                        
                        if self.args.video:
                            mix = torch.stack((blurs), dim=1)
                            mix[:, len(blurs)//2, :, :, :] = reblur
                            output = inner_model_obj.deblur_model(mix.detach())
                        else:
                            output = inner_model_obj.deblur_model(reblur.detach())
                 
                    # Calc loss
                    #*********************************************************** blurs -> sharps
                    loss = self.loss_all['deblur_loss'](output, blurs[len(blurs) // 2], reblur.detach())
                
                loss /= self.args.support_batch
                
                # Backward
                loss.backward()
                if (i+1) % self.args.support_batch == 0:
                    inner_optim.step()
                
                running_loss += loss.item()
                tq.set_postfix(loss=(running_loss/(i+1)))

        # print reblur quality
        if self.args.reblur_result:
            psnr_ave = psnr / len(data_save)
            ssim_ave = ssim / len(data_save)
            lpips_ave = lpips / len(data_save)
            self.logger.info('Support  PSNR (now): %.3f, \t SSIM (now): %.4f, \t LPIPS (now): %.4f' %(psnr_ave, ssim_ave, lpips_ave)) 
        
        self.meta_model_sd = copy.deepcopy(inner_model.state_dict())
        self.meta_gan_sd = copy.deepcopy(self.loss_all['gan_loss'].gan.vgg19.state_dict())

        return psnr_ave if self.args.reblur_result else 0

    def query(self, current_epoch=0):
        # reload the model weight after support set
        self.model.load_state_dict(self.meta_model_sd)
#         
        if self.args.meta_test or self.args.finetuning:
            ### Single image
            self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
           
            # Prepare map for whole img
            if not self.args.full_img_exp:
                self.W = int(math.ceil(self.args.img_w / self.args.input_w))
                self.H = int(math.ceil(self.args.img_h / self.args.input_h))

                pad_img = torch.zeros(1, 3, self.args.input_h*self.H, self.args.input_w*self.W)
                pad_gt = torch.zeros(1, 3, self.args.input_h*self.H, self.args.input_w*self.W)

            self.model.eval()
            with torch.no_grad():
                psnr, ssim, lpips, cnt = 0., 0., 0., 0
                
                tq = tqdm.tqdm(self.dataloader['test'], total=len(self.dataloader['test']), disable=(self.args.n_frames==1))
                tq.set_description('Evaluating...')
                
                for i, data_tup in enumerate(tq):
                    # Get patch position if not inference at full img size
                    if not self.args.full_img_exp:
                        pch_idx = int(i % (self.W * self.H))
                        row = int(pch_idx // self.W)
                        col = int(pch_idx - (row * self.W))
                   
                    # realblur dataset have diff size of img
                    if self.args.dataset == "RealBlur":
                        data, name, h, w = data_tup
                    else:
                        data, name = data_tup

                    # Fetch data
                    data = self.prepare(data)
                    blurs, sharps = data['blurs'], data['sharps']
                    
                    # Get deblur result
                    if self.args.video:
                        mix = torch.stack((blurs), dim=1)
                        output = self.model_obj.deblur_model(mix)
                    else:
                        output = self.model_obj.deblur_model(blurs[len(blurs)//2])

#                     output = self.model_obj.deblur_model(blurs[len(blurs) // 2])
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
                    
                    if not self.args.full_img_exp:
                        pad_img[:, :, row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w] = output[:, :, :self.args.input_h, :self.args.input_w]
                        pad_gt[:, :, row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w] = sharp[:, :, :self.args.input_h, :self.args.input_w]

                    ### calculate psnr and ssim
                    if self.args.full_img_exp or pch_idx == self.W*self.H - 1:
                        if not self.args.full_img_exp:
#                             full_img = pad_img[:, :, 0:self.args.img_h, 0:self.args.img_w]
#                             full_gt = pad_gt[:, :, 0:self.args.img_h, 0:self.args.img_w]
                            full_img = pad_img
                            full_gt = pad_gt
                        else:
                            full_img = output
                            full_gt = sharp
                        
                        if self.args.dataset == "RealBlur":
                            full_img = full_img[:, :, :h, :w]
                            full_gt = full_gt[:, :, :h, :w]

                        # Save image
                        if (self.args.save_results):
                            img_save = full_img
                            if self.args.centralized:
                                img_save = img_save + self.args.max_bound
                            if self.args.normalized:
                                img_save = (img_save / (self.args.max_bound - self.args.min_bound)) * 255
                            
                            img_save = np.transpose(img_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                            imsave(os.path.join(self.args.save_dir, 'save_results', name[0]+'.png'), img_save)
                
                        # Calc PSNR/SSIM
                        cnt += 1
                        _psnr, _ssim, _lpips = calc_psnr_and_ssim(self.args, full_img.detach(), full_gt.detach())
                        
                        psnr += _psnr
                        ssim += _ssim
                        lpips += _lpips

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                lpips_ave = lpips / cnt
                
                # Calc max PSNR/SSIm 
                if (psnr_ave > self.model_obj.max_psnr):
                    self.model_obj.max_psnr = psnr_ave
                    self.model_obj.max_psnr_epoch = current_epoch
                if (ssim_ave > self.model_obj.max_ssim):
                    self.model_obj.max_ssim = ssim_ave
                    self.model_obj.max_ssim_epoch = current_epoch
                ### Single image
                self.logger.info('Ref  PSNR (now): %.3f, \t SSIM (now): %.4f, LPIPS (now): %.4f' %(psnr_ave, ssim_ave, lpips_ave))

            self.logger.info('Evaluation over.')

            return psnr_ave, ssim_ave, lpips_ave, cnt
        else:
            # Set model
            self.model.train()
            if (not self.args.gan) and (not self.args.cycle_update) and (not self.args.reblur_backward):
                for param in self.model_obj.reblur_model.parameters():
                    param.requires_grad = False

            running_loss = 0
            
            tq = tqdm.tqdm(self.dataloader['test'], total=len(self.dataloader['test']))
            tq.set_description('Epoch {}, Video {}, lr {}'.format(current_epoch, self.cur_video, self.optimizer.param_groups[0]['lr']))
            
            for i, data_tup in enumerate(tq):
                if i % self.args.query_batch == 0:
                    if self.args.gan:
                        self.loss_all['gan_loss'].gan.vgg19.load_state_dict(self.meta_gan_sd)
                    self.optimizer.zero_grad()
                
                self.model.load_state_dict(self.meta_model_sd)

                # Query set
                loss = 0
                
                # Prepare data
                data, name = data_tup
                data = self.prepare(data)
                blurs, sharps = data['blurs'], data['sharps']
               
                # Update both reblur and deblur model
                if self.args.gan or self.args.cycle_update or self.args.reblur_backward:
                    
                    if self.args.combine_update:
                        # reblur loss
                        latent, output = self.model(blurs, sharps)
                        reblur_loss = self.loss_all['reblur_loss'](latent, blurs[len(blurs) // 2])
                    
                        # cycle loss
                        cycle_loss = self.loss_all['deblur_loss'](output, torch.repeat_interleave(sharps[len(sharps) // 2], 2, dim=0), blurs[len(blurs) // 2])
                    else:
                        # reblur loss
                        if self.args.reblur_model == "mtrnn":
                            X = torch.cat(sharps, dim=1)
                        elif self.args.reblur_model == "attn":
                            X = torch.cat([sharp.unsqueeze(dim=1) for sharp in sharps], dim=1)       # B F C H W
                        
                        latent = self.model_obj.reblur_model(X)
                        reblur_loss = self.loss_all['reblur_loss'](latent, blurs[len(blurs) // 2])

                        # cycle loss
                        if self.args.video:
                            mix = torch.stack((blurs), dim=1)
                            output = self.model_obj.deblur_model(mix)
                        else:
                            output = self.model_obj.deblur_model(blurs[len(blurs)//2])
#                         output = self.model_obj.deblur_model(blurs[len(blurs)//2])
                        cycle_loss = self.loss_all['deblur_loss'](output, sharps[len(sharps) // 2], blurs[len(blurs) // 2])

                    # gan loss
                    gan_loss = 0
#                     if self.args.gan:
#                         rand = random.randint(0, len(blurs)-1)
#                         while(rand == len(blurs) // 2):
#                             rand = random.randint(0, len(blurs)-1)
#                         gan_loss = self.loss_all['gan_loss'](math.nan, latent, blurs[rand])
                    
                    loss = (self.args.reblur_ratio * reblur_loss) + ((1 - self.args.reblur_ratio) * cycle_loss) + (0.1 * gan_loss)

                # Only update deblur model
                else:
                    if self.args.video:
                        mix = torch.stack((blurs), dim=1)
                        output = self.model_obj.deblur_model(mix)
                    else:
                        output = self.model_obj.deblur_model(blurs[len(blurs) // 2])
                    loss = self.loss_all['deblur_loss'](output, sharps[len(sharps) // 2], blurs[len(blurs) // 2])
                
                # Calc loss and backward
                loss /= len(tq)
                loss.backward()
                
                # Update query set every n data
                if (i+1) % self.args.query_batch == 0:
                    self.model.load_state_dict(self.cur_model_sd)
                    self.optimizer.step()
                    self.cur_model_sd = copy.deepcopy(self.model.state_dict())

                running_loss += loss.item()
                tq.set_postfix(loss=(running_loss))

            return 0
