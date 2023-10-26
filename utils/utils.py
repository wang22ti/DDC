import torch
import shutil
import os
import numpy as np
import random

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)

def save_checkpoint(args, state, is_best, save_freq):
    
    if is_best:
        filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, filename)
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
        print("Storing the best model")
    if state['epoch']%save_freq == 0 or state['epoch'] == 1:
        filename = f"{args.root_model}/{args.store_name}/{state['epoch']}_ckpt.pth.tar"
        print("[INFORMATION] Storing the model at ", filename)
        torch.save(state, filename)


class AverageMeter(object):
    
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


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        if len(target.size()) > 1:
            _, target = torch.max(target.data, 1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    factor = args.epochs // 200
    epoch = epoch + 1
    if epoch <= 5 * factor:
        lr = args.lr * epoch / 5
    elif epoch > 180 * factor:
        lr = args.lr * 0.0001
    elif epoch > 160 * factor:
        lr = args.lr * 0.01
    else:
        lr = args.lr
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    factor = args.epochs // 200
    if epoch > 160 * factor:
        rho = args.rho[1]
    else:
        rho = args.rho[0]

    for param_group in optimizer.param_groups:
        param_group['rho'] = rho

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
