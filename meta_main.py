import time
import yaml
import copy
import random
import shutil
import numpy as np
from option import args
from optim import get_optim_and_scheduler
from utils import mkExpDir
from dataset.dataloader import get_dataloader
from model import model
from loss.loss import get_loss_dict
from metaTrainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### check device
    assert (args.gpu_id != None or args.gpu_num > 0) or args.cpu, "No device specify!"
    assert (not (args.test and args.eval)) , "Can't do evaluation and test simultaneously!"
    assert (not (args.test and args.test_image == None)), "Missing testing image directory!"

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
    cur_time = time.strftime("%Y-%m-%d-%X", time.localtime())       
    print("Start time: {}".format(cur_time))
    print("Model save dir: {}".format(args.save_dir))
    
    args.program_name = os.path.basename(args.save_dir)

    ### make save_dir
    logger = mkExpDir(args)

    if args.meta_test or args.finetuning:
        args.num_epochs = 1
    
    ### model
    model, start_epoch = model.get_model(args)
    ori_model_sd = copy.deepcopy(model.state_dict())
    cur_model_sd = copy.deepcopy(model.state_dict())

    ### Optimizer and scheduler
    optim, scheduler = get_optim_and_scheduler(args, model)

    ### set video list
    if args.meta_train:
        videos = sorted(os.listdir(os.path.join(args.dataset_dir, 'train')))
    elif args.meta_test or args.finetuning:
        videos = sorted(os.listdir(os.path.join(args.dataset_dir, 'test')))
    
    if args.meta_train and args.finetuning:
        psnr_calc = np.zeros(args.finetune_update)
    
    cnt = 0
    max_psnr = 0
#     gradients = []
    for epoch in range(1, args.num_epochs + 1): 
        psnr = 0
        ssim = 0
        lpips = 0
        reblur_psnr = 0
        frame_sum = 0

        # random shuffle
        if args.meta_train and args.video_shuffle:
            random.shuffle(videos)
        
        # iterate all the video
        for idx, video in enumerate(videos):
            cnt += 1
            args.cur_video = video
            print("\nCurrent video: {}".format(video))

            # model load
            model.load_state_dict(ori_model_sd)
            
            ### loss
            loss_all = get_loss_dict(args)
    
            # dataloader of training set and testing set
            dataloader = get_dataloader(args)
            
            # init trainer
            t = Trainer(args, logger, dataloader, model, cur_model_sd, loss_all, optim, scheduler, video, idx)
            
            ### test / train
            if args.meta_train:
                _ = t.support(current_epoch=epoch)
                _ = t.query(current_epoch=epoch)
                
                cur_model_sd = model.state_dict()

                if cnt % args.task_batch_size == 0:
                    # reset model
                    scheduler.step()

                    # save last model weight
                    prtn_str = str(cnt // args.task_batch_size).zfill(5)
                    model_obj = model.module if ((not args.cpu) and (args.gpu_num > 1)) else model
                    model_obj.save_model(prtn_str)

                    if args.gan:
                        # save discriminator for GAN
                        tmp = loss_all['gan_loss'].gan.vgg19.state_dict()
                        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp}
                        model_name = args.save_dir.strip('/')+'/model/gan_'+str(epoch).zfill(5)+'.pt'
                        torch.save(model_state_dict, model_name)

                    # renew model state dict
                    ori_model_sd = copy.deepcopy(model.state_dict())

            else:
                start = time.process_time()
                r_psnr = t.support(current_epoch=epoch)
                end = time.process_time()
#                 print("Inner update elapse time {}s".format(end-start))

                _psnr, _ssim, _lpips, _cnt = t.query(current_epoch=epoch)
                psnr += _psnr * _cnt
                ssim += _ssim * _cnt
                lpips += _lpips * _cnt
                reblur_psnr += r_psnr
                frame_sum += _cnt
       
            torch.cuda.empty_cache()

        # calc all iter psnr of all video when finetuning
        if args.finetuning:
            psnr_calc[it] += psnr / frame_sum
            
        if args.meta_test:
#             print(max_psnr)
            logger.info('Avg PSNR: %.3f, Avg SSIM: %.3f, Avg LPIPS: %.3f' %(psnr/frame_sum, ssim/frame_sum, lpips/frame_sum))
            if args.reblur_result:
                logger.info('Avg Reblur PSNR: %.3f' %(reblur_psnr/len(videos)))
            if psnr / frame_sum > max_psnr:
                max_psnr = psnr / frame_sum
#             logger.info('Max PSNR: %.3f' %(max_psnr))
    
