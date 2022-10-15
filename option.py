import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='exp')

### log setting
parser.add_argument('--save_dir', type=str, default='save_dir',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='exp.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='EXP',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--gpu_num', type=int, default=1,
                    help='The number of GPU to be used')
parser.add_argument('--gpu_id', type=str, default=None,
                    help='The GPUs id to be used in training')

### dataset setting
parser.add_argument('--dataset', type=str, default='GOPRO',
                    help='Which dataset to train and test')
parser.add_argument('--dataset_dir', type=str, default='~/datasets/benchMark/GOPRO_Large/',
                    help='Directory of dataset')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=4,
                    help='The number of workers when loading data')
parser.add_argument('--img_w', type=int, default=1280,
                    help='The width of the img')
parser.add_argument('--img_h', type=int, default=720,
                    help='The height of the img')
parser.add_argument('--input_w', type=int, default=1280,
                    help='The width of the input patch')
parser.add_argument('--input_h', type=int, default=720,
                    help='The height of the input patch')
parser.add_argument('--normalized', type=str2bool, default=True,
                    help='Data range normalize to 1')
parser.add_argument('--centralized', type=str2bool, default=True,
                    help='Data range set to center')

### model setting 
parser.add_argument('--random_seed', type=int, default=43,
                    help='the random seed of the model')
parser.add_argument('--reblur_model', type=str, default='mtrnn',
                    help='the reblur model structure')
parser.add_argument('--deblur_model', type=str, default='swtn',
                    help='the deblur model structure')
parser.add_argument('--deblur_model_path', type=str, default=None,
                    help='the pretrained deblur model path for training or testing')
parser.add_argument('--reblur_model_path', type=str, default=None,
                    help='the pretrained reblur model path for training or testing')
parser.add_argument('--gan_model_path', type=str, default=None,
                    help='the pretrained model path for discriminator for GAN')
parser.add_argument('--video', type=str2bool, default=False,
                    help='deblur model is video deblurring or single image deblurring method')
parser.add_argument('--features', type=int, default=32,
                    help='The number of the channels in the reblur network')
parser.add_argument('--reblur_layers', type=int, default=3,
                    help='The number of the channels in the reblur network')
parser.add_argument('--n_frames', type=int, default=5,
                    help='The number of frames input to reblur model at a time')
parser.add_argument('--n_critics', type=int, default=1,
                    help='The number of batch that discriminator update every time')
parser.add_argument('--use_attn', type=str2bool, default=True,
                    help='use attention layer in reblurring model or 3d convolution')

### training setting
parser.add_argument('--batch_size', type=int, default=8,
                    help='Training batch size')
parser.add_argument('--rec_loss', type=str, default='l2',
                    help='reconstruction loss type, "l1" and "l2"')
parser.add_argument('--reblur_lr', type=float, default=1e-4,
                    help='Learning rate for training reblur model')
parser.add_argument('--deblur_lr', type=float, default=1e-4,
                    help='Learning rate for training deblur model')
parser.add_argument('--gan_lr', type=float, default=1e-4,
                    help='Learning rate for training gan discriminator model')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='The number of training epochs')
parser.add_argument('--optim', type=str, default='adam',
                    help='The optimizer during training')
parser.add_argument('--scheduler', type=str, default='stepLR',
                    help='The scheduler during training')
parser.add_argument('--decay', type=int, default=9999,
                    help='The step size of scheduler for decay during training')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='The gamma of scheduler during training')
parser.add_argument('--reblur_ratio', type=float, default=0.5,
                    help='The reblur loss ratio regarding to cycle loss (0-1)')
parser.add_argument('--alt_update', type=str2bool, default=False,
                    help='whether to alternative update deblur and reblur model')
parser.add_argument('--gan', type=str2bool, default=False,
                    help='whether to use GAN loss when training')
parser.add_argument('--gan_ratio', type=float, default=0.3,
                    help='gan loss ratio')
parser.add_argument('--reblur_result', type=str2bool, default=False,
                    help='whether to get reblur result in inference time')
parser.add_argument('--deblur_warnup', type=int, default=0,
                    help='whether to have warnup epoch when updating deblur model')

### meta learning setting
parser.add_argument('--inner_lr', type=float, default=1e-6,
                    help='Learning rate for meta inner loop')
parser.add_argument('--meta_lr', type=float, default=1e-7,
                    help='Learning rate for meta outer loop')
parser.add_argument('--use_fix_update', type=str2bool, default=True,
                    help='whether to use fix update or percentage base on video frame number')
parser.add_argument('--n_updates', type=int, default=30,
                    help='Number of support data patch to be used in both meta training and testing')
parser.add_argument('--cur_video', type=str, default=None,
                    help='the current video to be use when meta learning')
parser.add_argument('--video_shuffle', type=str2bool, default=True,
                    help='whether to shuffle video when meta training')
parser.add_argument('--meta', type=str2bool, default=False,
                    help='meta stage')
parser.add_argument('--meta_test', type=str2bool, default=False,
                    help='meta inference phase')
parser.add_argument('--meta_train', type=str2bool, default=False,
                    help='meta training phase')
parser.add_argument('--finetuning', type=str2bool, default=False,
                    help='finetuning mode')
parser.add_argument('--support_optim', type=str, default='adam',
                    help='the optimizer of support set')
parser.add_argument('--support_size', type=int, default=256,
                    help='the patch size use for support set')
parser.add_argument('--support_epochs', type=int, default=1,
                    help='The number of support set epochs')
parser.add_argument('--reblur_epochs', type=int, default=1,
                    help='The number of reblur model update epochs')
parser.add_argument('--support_batch', type=int, default=1,
                    help='support set update every n iterations')
parser.add_argument('--query_batch', type=int, default=1,
                    help='query set update every n iterations')
parser.add_argument('--task_batch_size', type=int, default=1,
                    help='how many task to backpropagate at a time')
parser.add_argument('--full_img_exp', type=str2bool, default=False,
                    help='whether to use full image as inference(query)')
parser.add_argument('--full_img_sup', type=str2bool, default=False,
                    help='whether to use full image as support set')
parser.add_argument('--use_reblur_pair', type=str2bool, default=True,
                    help='whether to use blur-reblur pair as support set in meta training')
parser.add_argument('--reblur_method', type=str, default="4x",
                    help='decide which reblur methods are be used (base, 4x, 2x, 4x_5)')
parser.add_argument('--use_blurrest', type=str2bool, default=False,
                    help='whether to use blurrest or neighbor patch as GAN real sample')
parser.add_argument('--cycle_update', type=str2bool, default=False,
                    help='whether to use cycle consistency loss when update reblur model')
parser.add_argument('--reblur_backward', type=str2bool, default=False,
                    help='whether to backward reblurs loss during support set')
parser.add_argument('--combine_update', type=str2bool, default=False,
                    help='whether to combine reblur and blur data at meta training query set')
parser.add_argument('--cycle_block', type=str2bool, default=False,
                    help='whether to block deblurring model to update by cycle loss')
parser.add_argument('--round', type=str2bool, default=False,
                    help='round when updating deblurring model')
parser.add_argument('--use_inner_lr', type=str2bool, default=False,
                    help='whether to use inner lr to update deblurring model')
parser.add_argument('--tile', type=str2bool, default=False,
                    help='whether to use tile patches for self-shift or random crop')
parser.add_argument('--find_sharp', type=str, default="self-shift",
                    help='what methods are used to find relative sharp patches w/o reference (among self-shift, niqe, brisque)')
parser.add_argument('--diff_method', type=str, default="psnr",
                    help='the method to be used for self-shift differce calculating')

### val/eval/test setting
parser.add_argument('--validate', type=str2bool, default='true',
                    help='Whether do validation or not')
parser.add_argument('--val_every', type=int, default=999999,
                    help='Validation period')
parser.add_argument('--save_every', type=int, default=999999,
                    help='Save period')
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')
parser.add_argument('--test_img', type=str, default=None,
                    help='The path of input testing image')
parser.add_argument('--save_results', type=str2bool, default=False,
                    help='Save each image during testing or validating')

args = parser.parse_args()
