from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    if args.meta_train or args.meta_test:
        m = import_module('dataset.' + 'meta_' + args.dataset.lower())
    else:
        m = import_module('dataset.' + args.dataset.lower())

    dataloader = {}
    
    if args.dataset == 'GOPRO' or args.dataset == "DVD" or args.dataset == "REDS" or args.dataset == "RealBlur":
        if not args.test and not args.eval:
            data_train = getattr(m, 'TrainSet')(args)
            dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            dataloader["train"] = dataloader_train
        if args.validate or args.eval or args.test or args.meta:
            data_test = getattr(m, 'TestSet')(args)
            dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=True)
            dataloader["test"] = dataloader_test

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader
