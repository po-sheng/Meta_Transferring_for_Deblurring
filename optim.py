import torch.optim as optim

def get_optim(args, model):
    if ((not args.cpu) and (args.gpu_num > 1)):
        reblur_params = model.module.reblur_model.parameters()
        deblur_params = model.module.deblur_model.parameters()
        # only calc gradient on conv part
#         params = [p for p in model.module.parameters() if p not in model.module.model.parameters()]

        deblur_lr = model.module.deblur_lr
        reblur_lr = model.module.reblur_lr
    else:
        reblur_params = model.reblur_model.parameters()
        deblur_params = model.deblur_model.parameters()
        # only calc gradient on conv part
#         params = [p for p in model.parameters() if p not in model.model.atten.parameters()]
        
        deblur_lr = model.deblur_lr
        reblur_lr = model.reblur_lr

    if args.optim == 'adam':
        optimizer = optim.Adam([{'params': reblur_params, 'lr': reblur_lr},
                                {'params': deblur_params}], lr=deblur_lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD([{'params': reblur_params, 'lr': reblur_lr},
                                {'params': deblur_params}], lr=deblur_lr, momentum=0.9)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta([{'params': reblur_params, 'lr': reblur_lr},
                                    {'params': deblur_params}], lr=deblur_lr)
    elif args.optim == 'adamW':
        optimizer = optim.AdamW([{'params': reblur_params, 'lr': reblur_lr},
                                    {'params': deblur_params}], lr=deblur_lr)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop([{'params': reblur_params, 'lr': reblur_lr},
                                    {'params': deblur_params}], lr=deblur_lr)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad([{'params': reblur_params, 'lr': reblur_lr},
                                    {'params': deblur_params}], lr=deblur_lr)
    else:
        raise ValueError("Optimizer {} not recognized.".format(args.optim))

    return optimizer

def get_scheduler(args, optimizer):
    if args.scheduler == 'stepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-8)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, threshold=1e-6, patience=10)
    else:
        raise ValueError("Scheduler {} not recognized.".format(args.scheduler))

    return scheduler

def get_optim_and_scheduler(args, model):
    optim = get_optim(args, model)
    scheduler = get_scheduler(args, optim)

    return optim, scheduler
