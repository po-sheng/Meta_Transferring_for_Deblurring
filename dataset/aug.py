import math
import torch
import random
import numpy as np


def expandPatch(Hpos, Wpos, h, w, patch_size):
    expand_patch = 4

    pch_h = Hpos[1] - Hpos[0]
    pch_w = Wpos[1] - Wpos[0]
    
    Hrange = [] 
    Wrange = []

    # use a large patch as ref 
    if Hpos[0] < pch_h:
        Hrange.append(0)
    elif Hpos[1] > h-pch_h:
        Hrange.append(h-(pch_h+(patch_size*expand_patch*2)))
    else:
        Hrange.append(Hpos[0]-(patch_size*expand_patch))

    if Wpos[0] < pch_w:
        Wrange.append(0)
    elif Wpos[1] > w-pch_w:
        Wrange.append(w-(pch_w+(patch_size*expand_patch*2)))
    else:
        Wrange.append(Wpos[0]-(patch_size*expand_patch))
    
    Hpos[0] = int(Hrange[0])
    Hpos[1] = int(Hrange[0] + pch_h + (2*expand_patch))
    Wpos[0] = int(Wrange[0])
    Wpos[1] = int(Wrange[0] + pch_w + (2*expand_patch))

    return Hpos, Wpos

class RandomRotate(object):
    def __call__(self, data):
        dirct = random.randint(0, 4)
        for key in data.keys():
            for i, frame in enumerate(data[key]):
                data[key][i] = np.rot90(frame, dirct).copy()

        return data

class Normalize(object):
#     def __init__(self, d_model="swtn", half2half=["swtn"], zero2one=[]):
    def __init__(self, center=True, norm=True):
        super(Normalize, self).__init__()
        self.center = center
        self.norm = norm

    def __call__(self, data):
        for key in data.keys():
            for i, frame in enumerate(data[key]):
                norm_img = frame
                if self.center:
                    norm_img = (norm_img - 255/2).copy() 
                if self.norm:
                    norm_img = (norm_img / 255).copy()
                data[key][i] = norm_img

        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 2) == 1:
            for key in data.keys():
                for i, frame in enumerate(data[key]):
                    data[key][i] = np.fliplr(frame).copy()
        if random.randint(0, 2) == 1:
            for key in data.keys():
                for i, frame in enumerate(data[key]):
                    data[key][i] = np.flipud(frame).copy()

        return data

class Resize(object):
    def __init__(self, img_h, img_w, factor=1):
        super(Resize, self).__init__()
        self._img_h = img_h
        self._img_w = img_w
        self._factor = factor

    def __call__(self, data):
        for key in data.keys():
            for i, frame in enumerate(data[key]):
                h = frame.shape[0]
                w = frame.shape[1]
                
                _img_h = math.ceil(h / self._factor) * self._factor
                _img_w = math.ceil(w / self._factor) * self._factor

                data[key][i] = np.pad(frame, ((0, _img_h - h), (0, _img_w - w), (0, 0)), mode='reflect').copy()
    
        return data

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0][0])
        h, w = self.Hsize, self.Wsize
    
        top = random.randint(0, H-h)
        left = random.randint(0, W-w)

        for key in data.keys():
            new_data = []
            for i, frame in enumerate(data[key]):
                data[key][i] = frame[top:top+h, left:left+w].copy()

        return data

class ToTensor(object):
    def __call__(self, data):
        new_data = {}   

        for key in data.keys():
            tmp = []
            for frame in data[key]:
                tmp.append(torch.from_numpy(frame.transpose((2, 0, 1))).float())
            new_data[key] = tmp

        return new_data
