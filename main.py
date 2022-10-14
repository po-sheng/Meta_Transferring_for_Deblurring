import time
import yaml
import random
import numpy as np
from option import args
from optim import get_optim_and_scheduler
from utils import mkExpDir
from dataset.dataloader import get_dataloader
from model import model
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### check device
    assert (args.gpu_id != None or args.gpu_num > 0) or args.cpu, "No device specify!"
    assert (not (args.test and args.eval)) , "Can't do evaluation and test simultaneously!"
    assert (not (args.test  == None)), "Missing testing image directory!"

    ### set args data range
    with open("model/setting.yaml") as f:
        models = yaml.safe_load(f)
    
    args.centralized = False
    args.normalized = False
    if args.deblur_model in models['data_range']['-0.5_0.5']:
        args.centralized = True
        args.normalized = True
        args.max_bound = 0.5
        args.min_bound = -0.5
    if args.deblur_model in models['data_range']['0_1']:
        args.normalized = True
        args.max_bound = 1
        args.min_bound = 0
    if args.deblur_model in models['data_range']['0_255']:
        args.max_bound = 255
        args.min_bound = 0

    ### set args model type
    if args.deblur_model in models['image_or_video']['video']:
        args.video = True
    else:
        args.video = False

    ### set gpu number  
    if args.gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        args.gpu_num = len((args.gpu_id).split(','))

    ### set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    ### args.save_dir
    time = time.strftime("%Y-%m-%d-%X", time.localtime())       
    print("Start time: {}".format(time))
    print("Model save dir: {}".format(args.save_dir))
    
    ### make save_dir
    logger = mkExpDir(args)

    ### dataloader of training set and testing set
    dataloader = get_dataloader(args)
    
    ### model
    model, start_epoch = model.get_model(args)

    ### loss
    loss_all = get_loss_dict(args)
    
    ### Optimizer and scheduler
    optim, scheduler = get_optim_and_scheduler(args, model)

    ### trainer
    t = Trainer(args, logger, dataloader, model, loss_all, optim, scheduler)
    
    ### test / train
    if (args.eval):     
        t.evaluate(start_epoch)
    elif (args.test):     
        t.evaluate(start_epoch)
    else:
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=(epoch+start_epoch))
            if ((epoch+start_epoch) % args.val_every == 0):
                t.evaluate(current_epoch=(epoch+start_epoch))
