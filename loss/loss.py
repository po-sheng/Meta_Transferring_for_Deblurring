import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from skimage.transform import pyramid_gaussian

### =======================================================================
### Basic loss 

class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, result, gt):
        return self.loss(result, gt)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, result, gt):
        return self.loss(result, gt)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, result, gt):
        diff = result - gt

        return torch.mean(torch.sqrt((diff*diff) + (self.eps*self.eps)))


# Hard Example Mining
class HEM(nn.Module):
    def __init__(self, hard_thre_p=0.5, device='cuda', random_thre_p=0.1):
        super(HEM, self).__init__()
        self.hard_thre_p = hard_thre_p
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()
        self.device = device

    def hard_mining_mask(self, x, y):
        with torch.no_grad():
            b, c, h, w = x.size()

            hard_mask = np.zeros(shape=(b, 1, h, w))
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
            res_numpy = res.cpu().numpy()
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)]
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind].item()
                hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

            random_thre_ind = int(self.random_thre_p * w * h)
            random_mask = np.zeros(shape=(b, 1 * h * w))
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                np.random.shuffle(random_mask[i])
            random_mask = np.reshape(random_mask, (b, 1, h, w))

            mask = hard_mask + random_mask
            mask = (mask > 0.).astype(np.float32)

            mask = torch.Tensor(mask).to(self.device)

        return mask

    def forward(self, x, y):
        mask = self.hard_mining_mask(x.detach(), y.detach()).detach()

        hem_loss = self.L1_loss(x * mask, y * mask)

        return hem_loss


### ===========================================================================
### Model oriented loss
class MTRNNLoss(nn.Module):
    def __init__(self):
        super(MTRNNLoss, self).__init__()
        self.loss = ReconstructionLoss('l1')

    def forward(self, result, gt):
        return self.loss(result, gt)


class CDVD_TSPLoss(nn.Module):
    def __init__(self):
        super(CDVD_TSPLoss, self).__init__()
        self.loss = ReconstructionLoss('l1')
        self.HEM = HEM()

    def forward(self, result, gt):
        return self.loss(result, gt) + 2*self.HEM()


class RestormerLoss(nn.Module):
    def __init__(self):
        super(RestormerLoss, self).__init__()
        self.loss = ReconstructionLoss('l1')

    def forward(self, result, gt):
        return self.loss(result, gt)


class MIMOUNetLoss(nn.Module):
    def __init__(self):
        super(MIMOUNetLoss, self).__init__()
        self.loss = ReconstructionLoss('l1')

    def forward(self, result, gt):
        # spatial L1
        loss = self.loss(result, gt)

        # spectrum L1
        result_fft = torch.fft.fft(result)
        gt_fft = torch.fft.fft(gt)
        loss += 0.1 * self.loss(result_fft, gt_fft)

        return loss


class MPRNetLoss(nn.Module):
    def __init__(self, lbd=0.05, eps=1e-3, contrastive=False):
        super(MPRNetLoss, self).__init__()
        self.eps = eps
        self.lbd = lbd
        self.contrastive = contrastive

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()

        self.charbonnier = CharbonnierLoss(eps=eps)

    '''
        def contrastive(self):
            self.vgg = Vgg19().cuda()
            self.l1 = nn.L1Loss()
            self.down_sample_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')

        def forward(self, restore, sharp, blur):
            B, C, H, W = restore.size()
            restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)
            threshold = 0.01
            mask = torch.mean(torch.abs(sharp-blur), dim=1).view(B, 1, H, W) # 0 ~ 1
            mask[mask <= threshold] = 0
            mask[mask > threshold] = 1
            mask = self.down_sample_4(mask)

            d_ap = torch.mean(torch.abs((restore_vgg - sharp_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
            d_an = torch.mean(torch.abs((restore_vgg - blur_vgg.detach())), dim=1).view(B, 1, H//4, W//4)

            mask_size = torch.sum(mask)
            contrastive = torch.sum((d_ap / (d_an + 1e-7)) * mask) / mask_size

            return contrastive
    '''

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        
        return diff

    def forward(self, result, gt):
        # Charbonnier loss
        char_loss = self.charbonnier(result, gt)

        # Edge loss
        edge_loss = self.charbonnier(self.laplacian_kernel(result), self.laplacian_kernel(gt))

        return char_loss + self.lbd * edge_loss


class GAN(nn.Module):
    def __init__(self, max_iter, lr, n_critics=1, decay=True, path=None):
        super(GAN, self).__init__()
        self.n_critics = n_critics
        self.decay = decay

        # Discriminator
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.classifier[6] = nn.Linear(4096, 1)
        if path != None:
            self.vgg19.load_state_dict(torch.load(path))
        self.vgg19 = nn.DataParallel(self.vgg19).cuda()

        self.vgg19.train()
        for param in self.vgg19.parameters():
            param.requires_grad = True
       
#         self.d_optim = torch.optim.Adam(self.vgg19.parameters(), lr=lr)
        self.d_optim = torch.optim.RMSprop(self.vgg19.parameters(), lr=lr)
        if self.decay:
            self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_optim, max_iter // self.n_critics, eta_min=1e-9)
    
        self.r_label = 1
        self.f_label = 0

    def forward(self, idx, fake, real):
        ### Update Discriminator
        if idx % self.n_critics == 0:
            # get result from discriminator
            real_output = self.vgg19(real.detach())
            fake_output = self.vgg19(fake.detach())
        
            self.d_optim.zero_grad()
            loss = -torch.mean(real_output) + torch.mean(fake_output)
            loss.backward()
            self.d_optim.step()
            if self.decay:
                self.d_scheduler.step()

            # clip weights of discriminator
            for p in self.vgg19.parameters():
                p.data.clamp_(-0.01, 0.01)

        ### Update Generator
        fake_output = self.vgg19(fake)

        return -torch.mean(fake_output)


### ==============================================================================
### Loss summary

class ReblurLoss(nn.Module):
    def __init__(self, reblur_model):
        super(ReblurLoss, self).__init__()
        self.reblur_model = reblur_model

        if reblur_model == "mtrnn" or reblur_model == "attn":
            self.loss = MPRNetLoss()
        else:
            raise ValueError("Reblur Model {} not recognized.".format(reblur_model))

    def forward(self, result, gt):
        return self.loss(result, gt)


class DeblurLoss(nn.Module):
    def __init__(self, deblur_model):
        super(DeblurLoss, self).__init__()
        self.deblur_model = deblur_model

        if deblur_model == "mprnet":
            self.loss = MPRNetLoss()
        elif deblur_model == "mimo" or deblur_model == "mimoPlus":
            self.loss = MIMOUNetLoss()
        elif deblur_model == "restormer":
            self.loss = RestormerLoss()
        elif deblur_model == "mtrnn":
            self.loss = MTRNNLoss()
        elif deblur_model == "cdvd_tsp":
            self.loss = CDVD_TSPLoss()
        else:
            raise ValueError("Deblur Model {} not recognized.".format(deblur_model))

    def forward(self, result, gt, blur, switch=2):
        if self.deblur_model == "mprnet":
            loss = 0
            for i in range(len(result)):
                loss += self.loss(result[i], gt)
            return loss
        
        elif self.deblur_model == "mimo" or self.deblur_model == "mimoPlus":
            loss = 0
            for i in range(len(result)):
                label = F.interpolate(gt, scale_factor=pow(0.5, len(result)-i-1), mode='bilinear')
                loss += self.loss(result[i], label)
            return loss
        
        elif self.deblur_model == "cdvd_tsp":
            assert len(gt) == 5, "CDVD_TSP should have 5 frame sequence as input!" 

            # past, current, and future
            recons_1, recons_2, recons_3, recons_2_iter, mid_loss = result
            result_list = torch.cat([recons_1, recons_2, recons_3, recons_2_iter], dim=1)
            gt_list = torch.cat([gt[1], gt[2], gt[3], gt[2]], dim=1)

            return self.loss(result_list, gt_list) + (mid_loss if mid_loss else 0)

        else:
            return self.loss(result, gt)

        return 0


class GanLoss(nn.Module):
    def __init__(self, n_updates, gan_lr, n_critics, meta=False, path=None):
        super(GanLoss, self).__init__()
        if meta:
            decay = False
        else:
            decay = True

        self.gan = GAN(n_updates, gan_lr, n_critics, decay=decay, path=path)

    def forward(self, idx, fake, real):
        return self.gan(idx, fake, real)

def get_loss_dict(args):
    loss = {}
    loss['reblur_loss'] = ReblurLoss(args.reblur_model)
    loss['deblur_loss'] = DeblurLoss(args.deblur_model)
    loss['gan_loss'] = GanLoss(args.n_updates, args.gan_lr, args.n_critics, meta=args.meta, path=args.gan_model_path)

    return loss
