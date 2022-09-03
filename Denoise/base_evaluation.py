import torch
from torch.autograd import Variable
import numpy as np
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from utils.base_module import adaptive
from visual_module.vutils import save_img
import os
import cv2
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def test(model, args, test_loader):
    ssim_test = AverageMeter('SIMM', ':6.2f')
    psnr_test = AverageMeter('PSNR', ':6.2f')
    progress = ProgressMeter(len(test_loader), [ssim_test, psnr_test], prefix='Test: ')
    #psnr_test = 0
    #ssim_test = 0
    model.eval()
    for batch in test_loader:
        noisy = Variable(batch[0].cuda())
        clean = Variable(batch[1].cuda())
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                prediction = model(noisy, clean)
                prediction = torch.clamp(prediction, 0., 1.)
        psnr = torch.Tensor([batch_PSNR(prediction, clean, 1.)]).cuda(args.local_rank)
        ssim = torch.Tensor([batch_SSIM(prediction, clean)]).cuda(args.local_rank) 
        torch.distributed.barrier()
        reduced_ssim = reduce_mean(ssim, args.workers)
        reduced_psnr = reduce_mean(psnr, args.workers)
        ssim_test.update(reduced_ssim.item(), noisy.size(0))
        psnr_test.update(reduced_psnr.item(), noisy.size(0))
        #torch.distributed.all_reduce(ssim, torch.distributed.ReduceOp.SUM)
        #torch.distributed.all_reduce(psnr, torch.distributed.ReduceOp.SUM)
    if args.local_rank == 0:
        print("===> Avg. PSNR", psnr_test.avg)
        print("===> Avg. SSIM", ssim_test.avg)
    return psnr_test.avg, ssim_test.avg


def test_decomposition(model, test_loader):
    psnr_test = 0
    ssim_test = 0
    model.eval()
    for batch in test_loader:
        noisy = Variable(batch[0].cuda())
        clean = Variable(batch[1].cuda())

        with torch.no_grad():
            reflectance, illumination, de_noise_reflectance, de_noise_illumination = model(noisy, 'noise')
            prediction = de_noise_reflectance * de_noise_illumination
            prediction = torch.clamp(prediction, 0., 1.)
        psnr_test += batch_PSNR(prediction, clean, 1.)
        ssim_test += batch_SSIM(prediction, clean)
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(test_loader)))
    print("===> Avg. SSIM: {:.4f}".format(ssim_test / len(test_loader)))
    return psnr_test / len(test_loader), ssim_test / len(test_loader)


def adaptive_test(model, test_loader, config, optimizer, criterion):
    psnr_test = 0
    ssim_test = 0
    model.eval()
    for batch in tqdm(test_loader):
        noisy = batch[0].cuda()
        clean = batch[1].cuda()
        prediction = adaptive(model, config.pretrain_path, noisy, criterion, optimizer)
        psnr_test += batch_PSNR(prediction, clean, 1.)
        ssim_test += batch_SSIM(prediction, clean)

    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(test_loader)))
    print("===> Avg. SSIM: {:.4f}".format(ssim_test / len(test_loader)))
    return psnr_test / len(test_loader), ssim_test / len(test_loader)


def adaptive_save_img(model, test_loader, config, optimizer, criterion, img, iteration):
    model.eval()
    base_path = r'E:\pipeline\image_denoising\image_denoise_pipeline\case'
    batch = test_loader.dataset[img]
    noisy = torch.from_numpy(batch[0]).cuda().unsqueeze(0)

    prediction = adaptive(model, config.pretrain_path, noisy, criterion, optimizer, iteration)
    save_img(prediction, os.path.join(base_path, 'nb2_%d_%d.PNG' % (img, iteration)))
    return 0


def batch_PSNR(img, imgClean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imgClean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def batch_SSIM(img, imgClean):
    Img = img.data.cpu().numpy().astype(np.float32)
    ImgClean = imgClean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for index in range(Img.shape[0]):
        for channel in range(Img.shape[1]):
            SSIM += structural_similarity(Img[index, channel, :, :].squeeze(), ImgClean[index, channel, :, :].squeeze())
    return SSIM / (Img.shape[0] * Img.shape[1])


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
