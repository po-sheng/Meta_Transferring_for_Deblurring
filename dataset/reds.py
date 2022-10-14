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
from .metaOperation import get_meta_data, self_shift
from .aug import expandPatch, RandomRotate, RandomFlip, Normalize, Resize, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args

        self.length = []                    # The # frame in each video
        self.blur_list = []
        self.sharp_list = []

        # Read all video frame path
        for video in os.listdir(os.path.join(args.dataset_dir, 'train')):
            if os.path.isdir(os.path.join(args.dataset_dir, 'train', video)):
                self.blur_list.append(sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "blur", '*.png'))))
                self.sharp_list.append(sorted(glob.glob(os.path.join(args.dataset_dir, 'train', video, "sharp", '*.png'))))
                self.length.append(len(self.blur_list[-1]) - (self.args.n_frames//2)*2)              # remove head and tail         
                
                assert len(self.blur_list[-1]) == len(self.sharp_list[-1]), "Video: {}, Missmatched Length!".format(video)
        
        self.transform = transforms.Compose([RandomCrop(args.input_h, args.input_w), RandomRotate(), RandomFlip(), Normalize(args.centralized, args.normalized), ToTensor()])

        assert (args.input_w % (4) == 0 and args.input_h % (4) == 0), "Image size should divided by patch size"

    def __len__(self):
        return sum(self.length)

    def __getitem__(self, idx):
        
        sharps = []
        blurs = []
        
        # Find video_idx and frame_idx
        for i in range(len(self.length)):
            if idx < self.length[i]:
                video_idx = i
                frame_idx = idx + (self.args.n_frames // 2) 
                break
            idx -= self.length[i]

        # Get a sequence of data
        for i in range(-1*(self.args.n_frames//2), (self.args.n_frames//2 + 1)):
            index = frame_idx + i
            
            ### read image
            sharps.append(imread(self.sharp_list[video_idx][index]).astype(np.float32))
            blurs.append(imread(self.blur_list[video_idx][index]).astype(np.float32))
        
        sample = {'sharps': sharps,  
                  'blurs': blurs}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args

        if not self.args.full_img_exp:
            self.W = int(math.ceil(self.args.img_w / self.args.input_w))
            self.H = int(math.ceil(self.args.img_h / self.args.input_h))
        
        self.length = []                        # The # frame in each video
        self.blur_list = []
        self.sharp_list = []
        
        # Read all video frame path
        for video in os.listdir(os.path.join(args.dataset_dir, 'test')):
            if os.path.isdir(os.path.join(args.dataset_dir, 'test', video)):
                self.blur_list.append(sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "blur", '*.png'))))
                self.sharp_list.append(sorted(glob.glob(os.path.join(args.dataset_dir, 'test', video, "sharp", '*.png'))))
                self.length.append(len(self.blur_list[-1]))                     # Test time dont need remove head and tail         
                
                assert len(self.blur_list[-1]) == len(self.sharp_list[-1]), "Video: {}, Missmatched Length!".format(video)
        
        if self.args.full_img_exp:
            self.transform = transforms.Compose([Normalize(args.centralized, args.normalized), Resize(self.args.img_h, self.args.img_w, 32), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(args.centralized, args.normalized), Resize(self.args.input_h, self.args.input_w, 32), ToTensor()])
        
    def __len__(self):
        if self.args.full_img_exp:
            return sum(self.length)
        else:
            return sum(self.length) * self.W * self.H

    def __getitem__(self, idx):
        
        sharps = []
        blurs = []
        
        # If using crop
        if not self.args.full_img_exp:
            pch_idx = idx % (self.H * self.W)    
            idx = idx // (self.H * self.W)
            row = pch_idx // self.W
            col = pch_idx % self.W

        # Find video_idx and frame_idx
        for i in range(len(self.length)):
            if idx < self.length[i]:
                video_idx = i
                frame_idx = idx 
                break
            idx -= self.length[i]
       
        # Get sequence data
        for i in range(-1*(self.args.n_frames//2), (self.args.n_frames//2 + 1)):
            index = frame_idx + i
            if index < 0:
                index = 0
            elif index >= self.length[video_idx]:
                index = self.length[video_idx] - 1
            
            sharp = imread(self.sharp_list[video_idx][index]).astype(np.float32)
            blur = imread(self.blur_list[video_idx][index]).astype(np.float32)
            
            if self.args.full_img_exp:
                sharps.append(sharp)
                blurs.append(blur)
            else:
                sharps.append(sharp[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
                blurs.append(blur[row*self.args.input_h:(row+1)*self.args.input_h, col*self.args.input_w:(col+1)*self.args.input_w])
        
        base, fname = os.path.split(self.sharp_list[video_idx][frame_idx]) 
        fname = fname.split(".")[0]
        base, _ = os.path.split(base)
        root, video = os.path.split(base)

        sample = {'sharps': sharps,  
                  'blurs': blurs}

        if self.transform:
            sample = self.transform(sample)

        name = video+"_"+fname
        
        return sample, name
