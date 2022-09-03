from __future__ import print_function
import os
import torch.distributed as dist
import pandas as pd
import argparse
import copy
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data import build_dataset
from base_train import train
from base_evaluation import test
from DMGNet import DMGNet
import math
import shutil
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
import torch
from timm.utils import NativeScaler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def init_parse():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Training settings
    parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
    parser.add_argument('--batch_size', type=int, default=15, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate. default=0.0001')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='if adopt augmentation when training')
    parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR', help='the training dataset')
    parser.add_argument('--Ispretrained', type=bool, default=True, help='If load checkpoint model')
    parser.add_argument("--noiseL", type=list, default=25, help='noise level')
    parser.add_argument('--save_folder', default='/home/ma-user/work/hanyudong/DeRain/DMGNet/checkpoint', help='Location to save checkpoint models')
    parser.add_argument('--save_pretrain', type=str, default='', help='the sub path to save pretrained model')

    # Testing settings
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')
    parser.add_argument('--seed', type=int, default=1111, help='random seed to use. Default=123')

    # Global settings
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--gpus', default=3, type=int, help='number of gpus')
    parser.add_argument('--workers', default=3, type=int, help='number of workers')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='the dataset dir')
    parser.add_argument('--sidd_path', default=r'/home/ma-user/work/hanyudong/data/Derain_train/Rain13K', help='sidd path')
    parser.add_argument('--testdata_path', default=r'/home/ma-user/work/hanyudong/data/Derain_test/Rain100H', help='sidd path')
    #parser.add_argument('--model_type', type=str, default='unet', help='the name of model')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped denoising image')
    parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')

    # Distributed Training
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--resume', action='store_true', default=False, help='load the log')
    args = parser.parse_args()
    return args

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2), # 转置卷积
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

def transfer_weights(new, old):
    model_dict = copy.deepcopy(new.module.state_dict()) # 只有一份参数
    for key, item in model_dict.items():
        model_dict[key] = old[key] # 取其中的一份参数
    return model_dict

def main():
    config = init_parse()
    check_dir = os.path.join(config.save_folder, config.save_pretrain)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    check_path = os.path.join(check_dir, 'best.pth')

    if config.resume:
        logs = torch.load(check_path)
        print("loading checkpoint...")

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(config.local_rank)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    print('===> Loading datasets')
    train_loader, val_loader = build_dataset(config) # 构建数据集
    model = DMGNet(k=4, img_size=128, embed_dim=48, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], win_size=8, mlp_ratio=4., qkv_bias=True, downsample=Downsample, upsample=Upsample)    
    model = model.cuda(config.local_rank)
    # 模型只有一份参数
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)
    criterion = CharbonnierLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.nEpochs-3, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
    scheduler.step()
    # save the model
    max_psnr = 0.0
    if config.resume:
        model.module.load_state_dict(transfer_weights(model, logs['model_state']))
        optimizer.load_state_dict(logs['optim_state'])
        config.start_iter = logs['epoch']
        max_psnr = logs['best_value'][0]
        ssim = logs['best_value'][1] 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.nEpochs-config.start_iter+1, eta_min=1e-6)
    #psnr, ssim = test(model, config, val_loader)
    #print(psnr, ssim)
    psnr_list = []
    ssim_list = []
    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()
    for epoch in range(config.start_iter, config.nEpochs + 1):
        train(loss_scaler, epoch, config, model, train_loader, optimizer, scheduler, criterion)
        psnr, ssim = test(model, config, val_loader)
        if psnr > max_psnr: # 只有更大才保存模型
            max_psnr = psnr
            results = {
                'epoch': epoch+1,
                'best_value': (max_psnr, ssim),
                'model_state': model.module.state_dict(),
                'optim_state': optimizer.state_dict(),
            }
            if config.local_rank == 0:
                torch.save(results, check_path)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        data_frame = pd.DataFrame(data={'epoch': epoch, 'PSNR': psnr_list, 'SSIM': ssim_list})
        data_frame.to_csv(os.path.join(check_dir, 'training_logs.csv'))
        # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (10) == 0:
            for param_group in optimizer.param_groups:
                print('Learning rate decay: lr={}'.format(param_group['lr']))


if __name__ == '__main__':
    main()
