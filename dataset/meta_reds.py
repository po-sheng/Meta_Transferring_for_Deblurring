import os
import copy
import math
from imageio import imread
import numpy as np
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from utils import calc_psnr, calc_ssim
from .metaOperation import find_sharp
from .aug import expandPatch, RandomRotate, RandomFlip, Normalize, Resize, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        video = args.cur_video
       
        self.W = int(math.ceil(self.args.img_w / self.args.support_size))
        self.H = int(math.ceil(self.args.img_h / self.args.support_size))
        
        if self.args.meta_train:
            if os.path.isdir(os.path.join(args.dataset_dir, 'train', video)):
                self.blur_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "blur", '*.png')))
                self.sharp_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "sharp", '*.png')))
        else:
            if os.path.isdir(os.path.join(args.dataset_dir, 'test', video)):
                self.blur_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "blur", '*.png')))
                self.sharp_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "sharp", '*.png')))

#         self.frame_self_shift = [self_shift(imread(self.blur_list[i]).astype(np.float32)) for i in range(self.args.n_frames//2, len(self.blur_list)-(self.args.n_frames//2))]
#         self.frame_self_shift_idx = np.argsort(self.frame_self_shift)
#         self.frame_self_shift_idx = self.frame_self_shift_idx[:(len(self.blur_list)*10)//100] + self.args.n_frames//2
#         random.shuffle(self.frame_self_shift_idx)

        if args.full_img_sup:
            if self.args.use_fix_update:
                self.support_idx = find_support_idx(self.blur_list, (self.args.n_updates * self.args.support_batch))
            else:
                self.support_idx = find_support_idx(self.blur_list, (self.args.n_updates * len(self.blur_list)) // 100)
        
        self.transform = transforms.Compose([Normalize(args.centralized, args.normalized), ToTensor()])

        assert len(self.blur_list) == len(self.sharp_list), "Missmatched Length!"
        assert (args.input_w % (4) == 0 and args.input_h % (4) == 0), "Image size should divided by patch size"

    def __len__(self):
        data_num = self.args.n_updates 
        if self.args.full_img_sup:
            data_num *= self.H * self.W

        if self.args.use_fix_update:
            data_num *= self.args.support_batch
        else:
            data_num = (data_num * len(self.blur_list)) // 100

        return data_num

    def __getitem__(self, idx):
        # Get sharpest frame and crop it into patch
        if self.args.full_img_sup:
            img_idx = idx // (self.H * self.W)
            row = (idx % (self.H * self.W)) // self.W
            col = (idx % (self.H * self.W)) % self.W

            idx = self.support_idx[img_idx]
            t, b, l, r = find_support_pos(row, col, self.args.support_size, self.args.img_h, self.args.img_w)
        
        # Get sharpest patch among all frames
        else:
            if self.args.tile:
                row = (idx % (self.H * self.W)) // self.W
                col = (idx % (self.H * self.W)) % self.W
                
                H = self.args.support_size * row if self.args.support_size * (row+1) < self.args.img_h else self.args.img_h - self.args.support_size
                W = self.args.support_size * col if self.args.support_size * (col+1) < self.args.img_w else self.args.img_w - self.args.support_size
            else:
                # Random crop patch
                H = random.randint(0, self.args.img_h - self.args.support_size)
                W = random.randint(0, self.args.img_w - self.args.support_size)

            # Find sharpest patch amoung all but head and tail frames
            min_diff, min_idx = float('inf'), 0
            max_diff, max_idx = float('-inf'), 0
            # Iter all frame except head and tail
            for i in range(self.args.n_frames//2, len(self.blur_list)-(self.args.n_frames//2)):
                img = imread(self.blur_list[i]).astype(np.float32)
                diff = find_sharp(img[H:H + self.args.support_size, W:W + self.args.support_size, :], method=self.args.find_sharp, diff_method=self.args.diff_method)
#                 print(i, diff) 
                # higher is blurrer (self-shift)
                if diff > max_diff:
                    max_diff = diff
                    max_idx = i
                
                # lower is sharper (self-shift)
                if diff < min_diff:
                    min_diff = diff
                    min_idx = i
            
            idx = max_idx if (self.args.meta_train and not self.args.use_reblur_pair) else min_idx
#             idx = self.frame_self_shift_idx[idx % len(self.frame_self_shift_idx)]
#             idx = random.randint(self.args.n_frames//2, len(self.blur_list)-(self.args.n_frames//2)-1)
            blur_idx = max_idx 

        sharps = []
        blurs = []
        blurrers = []
        
        for i in range(-1*(self.args.n_frames//2), (self.args.n_frames//2 + 1)):
            index = idx + i
            
            ### read image
            if self.args.full_img_sup:
                sharps.append(imread(self.sharp_list[index]).astype(np.float32)[t:b, l:r, :])
                blurs.append(imread(self.blur_list[index]).astype(np.float32)[t:b, l:r, :])
            else:
                sharps.append(imread(self.sharp_list[index]).astype(np.float32)[H:H + self.args.support_size, W:W + self.args.support_size, :])
                blurs.append(imread(self.blur_list[index]).astype(np.float32)[H:H + self.args.support_size, W:W + self.args.support_size, :])
       
        ### find blurrest patch
        if self.args.use_blurrest:
            if self.args.full_img_sup:
                blurrers.append(imread(self.blur_list[blur_idx]).astype(np.float32)[t:b, l:r, :])
            else:
                blurrers.append(imread(self.blur_list[blur_idx]).astype(np.float32)[H:H + self.args.support_size, W:W + self.args.support_size, :])
        
        sample = {'sharps': sharps,  
                  'blurs': blurs,
                  'blurrers': blurrers}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        video = args.cur_video
        
        self.W = int(math.ceil(self.args.img_w / self.args.input_w))
        self.H = int(math.ceil(self.args.img_h / self.args.input_h))
    
        if self.args.meta_train:
            if os.path.isdir(os.path.join(args.dataset_dir, 'train', video)):
                self.blur_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "blur", '*.png')))
                self.sharp_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "sharp", '*.png')))
        else:
            if os.path.isdir(os.path.join(args.dataset_dir, 'test', video)):
                self.blur_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "blur", '*.png')))
                self.sharp_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "sharp", '*.png')))

        if self.args.meta_train:
            if args.full_img_exp:
                self.transform = transforms.Compose([Normalize(args.centralized, args.normalized), ToTensor()])
            else:
                self.transform = transforms.Compose([RandomCrop(args.input_h, args.input_w), Normalize(args.centralized, args.normalized), ToTensor()])
        else:
            self.transform = transforms.Compose([Resize(self.args.input_h, self.args.input_w, 32), Normalize(args.centralized, args.normalized), ToTensor()])
        
        assert len(self.blur_list) == len(self.sharp_list), "Missmatched Length!"

    def __len__(self):
        if (self.args.meta_train and self.args.full_img_exp) or self.args.meta_test or self.args.finetuning:
            return len(self.sharp_list) * self.W * self.H
        else:
            return len(self.sharp_list)

    def __getitem__(self, idx):
        sharps = []
        blurs = []
        reblurs = []
        
        if (self.args.meta_train and self.args.full_img_exp) or self.args.meta_test or self.args.finetuning:
            img_idx = int(idx // (self.W * self.H))
            pch_idx = idx - img_idx * self.W * self.H

            row = int(pch_idx // self.W) 
            col = pch_idx - row*self.W
            
            idx = img_idx

        ### read image
        for i in range(-1*(self.args.n_frames//2), (self.args.n_frames//2 + 1)):
            index = idx + i
            if index < 0:
                index = 0
            elif index >= len(self.blur_list):
                index = len(self.blur_list) - 1
            
            if (self.args.meta_train and self.args.full_img_exp) or self.args.meta_test or self.args.finetuning:
                sharp = imread(self.sharp_list[index]).astype(np.float32)
                sharps.append(sharp[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
                blur = imread(self.blur_list[index]).astype(np.float32)
                blurs.append(blur[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
            
            else:
                sharps.append(imread(self.sharp_list[index]).astype(np.float32))
                blurs.append(imread(self.blur_list[index]).astype(np.float32))
        
#         name = self.sharp_list[idx]
        base, fname = os.path.split(self.sharp_list[idx]) 

        fname = fname.split(".")[0]
        base, _ = os.path.split(base)
        root, video = os.path.split(base)

        sample = {'sharps': sharps,  
                  'blurs': blurs}

        if self.transform:
            sample = self.transform(sample)

        name = video+"_"+fname
        
        return sample, name
