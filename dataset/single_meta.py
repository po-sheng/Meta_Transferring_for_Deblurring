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
from .metaOperation import self_shift, find_support_idx, find_support_pos
from .aug import expandPatch, RandomRotate, RandomFlip, Normalize, Resize, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.sharp, self.blur = args.cur_video
       
        self.crdinate = []
        score = []
#         for i in range(50):
#             H = random.randint(0, self.args.img_h - self.args.support_size)
#             W = random.randint(0, self.args.img_w - self.args.support_size)
#             
#             img = imread(self.blur).astype(np.float32)
#             score.append(self_shift(img[H:H + self.args.support_size, W:W + self.args.support_size, :]))
#             self.crdinate.append((H, W))

#         sort_score = np.argsort(score)
#         self.r_sharp_idx = copy.deepcopy(sort_score[:self.args.n_updates])
#         self.r_blur_idx = copy.deepcopy(sort_score[-self.args.n_updates:])

#         random.shuffle(self.r_sharp_idx)
#         random.shuffle(self.r_blur_idx)

        self.transform = transforms.Compose([Normalize(args.centralized, args.normalized), ToTensor()])

        assert (args.input_w % (4) == 0 and args.input_h % (4) == 0), "Image size should divided by patch size"

    def __len__(self):
        return self.args.n_updates

    def __getitem__(self, idx):
        sharps = []
        blurs = []
        blurrers = []
       
        H = random.randint(0, self.args.img_h - self.args.support_size)
        W = random.randint(0, self.args.img_w - self.args.support_size)
#         H, W = self.crdinate[self.r_sharp_idx[idx]]
#         blur_H, blur_W = self.crdinate[self.r_blur_idx[idx]]

        sharps.append(imread(self.sharp).astype(np.float32)[H:H + self.args.support_size, W:W + self.args.support_size, :])
        blurs.append(imread(self.blur).astype(np.float32)[H:H + self.args.support_size, W:W + self.args.support_size, :])
#         blurrers.append(imread(self.blur).astype(np.float32)[blur_H:blur_H + self.args.support_size, blur_W:blur_W + self.args.support_size, :])

        sample = {'sharps': sharps,  
                  'blurs': blurs,
                  'blurrers': blurrers}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.sharp, self.blur = args.cur_video
        
        self.W = int(math.ceil(self.args.img_w / self.args.input_w))
        self.H = int(math.ceil(self.args.img_h / self.args.input_h))
    
        if self.args.meta_train:
            self.transform = transforms.Compose([RandomCrop(args.input_h, args.input_w), Normalize(args.centralized, args.normalized), ToTensor()])
        else:
            self.transform = transforms.Compose([Resize(self.args.input_h, self.args.input_w, 32), Normalize(args.centralized, args.normalized), ToTensor()])
        
    def __len__(self):
        if self.args.meta_train:
            return self.args.n_updates
        elif self.args.meta_test:
            return 1

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
        if (self.args.meta_train and self.args.full_img_exp) or self.args.meta_test or self.args.finetuning:
            sharp = imread(self.sharp).astype(np.float32)
            sharps.append(sharp[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
            blur = imread(self.blur).astype(np.float32)
            blurs.append(blur[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
        
        else:
            sharps.append(imread(self.sharp).astype(np.float32))
            blurs.append(imread(self.blur).astype(np.float32))
        
#         name = self.sharp_list[idx]
        base, fname = os.path.split(self.sharp) 

        fname = fname.split(".")[0]
        base, _ = os.path.split(base)
        root, video = os.path.split(base)

        sample = {'sharps': sharps,  
                  'blurs': blurs}

        if self.transform:
            sample = self.transform(sample)

        name = video+"_"+fname
        
        return sample, name
