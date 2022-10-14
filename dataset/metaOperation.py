import os
import time
import copy
import numpy as np
from imageio import imread
import torch
import torch.nn as nn

from utils import calc_psnr

def reblur_method(method, img_ids, img_list):
    reblur = imread(img_list[img_ids]).astype(np.float32)
    
    if method == "base":
        for n in [-1, 1]:
            idx = img_ids + n
            if idx < 0:
                idx = 0
            elif idx >= len(img_list):
                idx = len(img_list) - 1
            reblur += imread(img_list[idx]).astype(np.float32)
        reblur /= 3
        
    elif method == "4x":
        base, fname_f = os.path.split(img_list[img_ids])
        base, _ = os.path.split(base)
        sp_fname_f = fname_f.split(".")
        if img_ids != len(img_list) - 1:
            _, fname_b = os.path.split(img_list[img_ids+1])
            sp_fname_b = fname_b.split(".")

        f = os.path.join(base, "reblur", sp_fname_f[0]+"_75."+sp_fname_f[1])
        if img_ids == 0:
            f = img_list[img_ids] 
        if img_ids == len(img_list) - 1:
            b = img_list[img_ids]
        else:
            b = os.path.join(base, "reblur", sp_fname_b[0]+"_25."+sp_fname_b[1])
    
        reblur += imread(f).astype(np.float32) + imread(b).astype(np.float32)
        reblur /= 3

    elif method == "2x":
        base, fname_f = os.path.split(img_list[img_ids])
        base, _ = os.path.split(base)
        sp_fname_f = fname_f.split(".")
        if img_ids != len(img_list) - 1:
            _, fname_b = os.path.split(img_list[img_ids+1])
            sp_fname_b = fname_b.split(".")

        f = os.path.join(base, "reblur", sp_fname_f[0]+"_50."+sp_fname_f[1])
        if img_ids == 0:
            f = img_list[img_ids] 
        if img_ids == len(img_list) - 1:
            b = img_list[img_ids]
        else:
            b = os.path.join(base, "reblur", sp_fname_b[0]+"_50."+sp_fname_b[1])
    
        reblur += imread(f).astype(np.float32) + imread(b).astype(np.float32)
        reblur /= 3

    elif method == "4x_5":
        base, fname_f = os.path.split(img_list[img_ids])
        base, _ = os.path.split(base)
        sp_fname_f = fname_f.split(".")
        if img_ids != len(img_list) - 1:
            _, fname_b = os.path.split(img_list[img_ids+1])
            sp_fname_b = fname_b.split(".")

        f_1 = os.path.join(base, "reblur", sp_fname_f[0]+"_75."+sp_fname_f[1])
        f_2 = os.path.join(base, "reblur", sp_fname_f[0]+"_50."+sp_fname_f[1])
        if img_ids == 0:
            f_1 = img_list[img_ids] 
            f_2 = img_list[img_ids] 
        if img_ids == len(img_list) - 1:
            b_1 = img_list[img_ids]
            b_2 = img_list[img_ids]
        else:
            b_1 = os.path.join(base, "reblur", sp_fname_b[0]+"_25."+sp_fname_b[1])
            b_2 = os.path.join(base, "reblur", sp_fname_b[0]+"_50."+sp_fname_b[1])
    
        reblur += imread(f_1).astype(np.float32) + imread(f_2).astype(np.float32) + imread(b_1).astype(np.float32) + imread(b_2).astype(np.float32)
        reblur /= 5
    
    return reblur


def find_support_idx(blur_list, num, order=0):
    ### if order == 0, choose the frame with clearest scene ###
    
    diff_list = []
    for path in blur_list:
        blur = imread(path).astype(np.float32)
        diff_list.append(self_shift(blur))
    diff = np.asarray(diff_list)

    sort_diff = np.argsort(diff)
    if order == 1:
        sort_diff = sort_diff[::-1]

    return sort_diff[:num]


def find_support_pos(row, col, patch, img_h, img_w):
    t, b, l, r = 0, 0, 0, 0

    if (row + 1) * patch <= img_h:
        t = row * patch
        b = (row + 1) * patch
    else:
        t = img_h - patch
        b = img_h

    if (col + 1) * patch <= img_w:
        l = col * patch
        r = (col + 1) * patch
    else:
        l = img_w - patch
        r = img_w

    return t, b, l, r    


def self_shift(img, diff_method="psnr"):
    # offset from four axis
    # rightTop-leftBot, y axis, rightBot-leftTop, x axis
    aspects = [(1, 0, 0, 1), (0, 0, 0, 1), (0, 1, 0, 1), (0, 1, 0, 0)]
    diff_sum = 0
    h, w, c = img.shape

    # calc ssim between img and its shift version
    for aspect in aspects:
        X_s, X_t, Y_s, Y_t = aspect
        if diff_method == "psnr":
            diff_sum += calc_psnr(img[Y_s:h-Y_t, X_s:w-X_t], img[Y_t:h-Y_s, X_t:w-X_s])
        elif diff_method == "minus":
            diff_sum += np.average(abs(img[Y_s:h-Y_t, X_s:w-X_t] - img[Y_t:h-Y_s, X_t:w-X_s]))
    
    return diff_sum


def find_sharp(img, method="self-shift", diff_method="psnr"):
    if method == "self-shift":
        return self_shift(img, diff_method=diff_method)
    elif method == "niqe":
        return Niqe(img)
    elif method == "brisque":
        return Brisque(img)
    else:
        raise ValueError("Method {} not recognized.".format(method))


def reblur_center(blurs):
    mid = len(blurs) // 2
    for i, frame in enumerate(blurs):
        if i != mid:
            blurs[mid] += blurs[i]

    blurs[mid] /= len(blurs)

#     for blur in blurs:

    return copy.deepcopy(blurs)


def reblur(blurs):
    avgPool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    newBlurs = []
    for i, blur in enumerate(blurs):
        newBlurs.append(avgPool(blur))

    return newBlurs

def get_pseudo_sharp(blur):
    
    return blur.copy()

def get_pseudo_blur(blur_list, idx, limits, blur_range, blur, flag):
    output = []
    tmp = imread(blur_list[idx])
    for i in range(len(blur)):
        if flag:
            img = np.zeros(blur[0].shape)
        else:
            img = np.zeros(tmp.shape)
        for j in range(blur_range):
            index = idx + i + j - (len(blur) // 2) - (blur_range // 2) ## -2 is for sequence index i
            if index < limits[0]:
                index =  limits[0]
            elif index >= limits[1]:
                index = limits[1] - 1

            img += imread(blur_list[index]).astype(np.float32)

        output.append(img / blur_range)

    return blur.copy()
#     return output

def get_meta_data(blur_list, idx, limits, blur, blur_range=5, flag=False):
    
    return get_pseudo_sharp(blur), get_pseudo_blur(blur_list, idx, limits, blur_range, blur, flag)


