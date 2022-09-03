import torch
from torch.autograd import Variable
import os
import time
from tqdm import tqdm
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

def train(loss_scaler, epoch, args, model, train_loader, optimizer, criterion):
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    cnt = 0
    for batch in train_loader:
        if cnt % 100 == 0 and args.local_rank == 0:
            print("-------------------", cnt)
        noisy = Variable(batch[0].cuda())
        clean = Variable(batch[1].cuda()) # ground truth
        model.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            prediction = model(noisy, clean)
            # Optimized Scheme
            res_loss = criterion(prediction, clean) / (clean.size()[0] * 2)
            #grad_loss = criterion(pred_map, gt_map) / (pred_map.size()[0])
            loss = res_loss #+ 0.5 * grad_loss
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.workers)
        losses.update(reduced_loss.item(), noisy.size(0))
        loss_scaler(loss, optimizer, parameters=model.parameters())
        #grad_optimizer.zero_grad()
        #loss_scaler(0.5 * grad_loss, grad_optimizer, parameters = param)
        cnt = cnt + 1
    if args.local_rank == 0:
        print("===> Epoch {} Complete: Avg.".format(epoch), losses)
