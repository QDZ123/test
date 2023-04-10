import argparse
import os
import shutil
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import densenet as dn
import pandas as pd
from os import listdir

import cv2

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import splitfolders

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of rawdata channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0


class DataProcess(Dataset):
    def __init__(self, path):
        self.data_info = []
        fileList = listdir(path)  # 获取当前文件夹下所有文件
        for child_dir in fileList:   # 类别名称
            digit = child_dir
            child_path = os.path.join(path, child_dir) # 小类别目录
            img_list = listdir(child_path)   # 小目录下所有文件
            n = len(img_list)
            for i in range(n):
                img_name = img_list[i]
                self.data_info.append((child_path + '/' + img_name, digit))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        filepath, label = self.data_info[index]
        data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # padding大小
        row, col = data.shape
        _max = max(col, row)
        temp = np.zeros((_max, _max), np.uint8)
        temp[0:row, 0:col] = data
        # resize
        # temp = cv2.dnn.blobFromImage(temp, 1 / 255.0, (32, 32), crop=False)
        # img = temp[0]

        # img = cv2.resize(temp, (64, 64), interpolation=cv2.INTER_CUBIC)
        # img = img.reshape(1, 64, 64)
        img = cv2.resize(temp, (48, 48), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(1, 48, 48)
        # print("img:", img)
        # resize = transforms.Resize([32, 32])
        # temp = resize(temp)
        return img, label


def main():
    global args, best_prec1
    args = parser.parse_args()
    # if args.tensorboard: configure("runs/%s" % (args.name))

    # 切分数据集
    # splitfolders.ratio('./rawdata', output="./split_data_raw", seed=1337, ratio=(0.8, 0.2))

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # 加载数据集

    trainingData = DataProcess("./split_data_raw_siam_2/train")
    testData = DataProcess('./split_data_raw_siam_2/val')

    train_loader = DataLoader(trainingData, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=True)

    # create model
    model = dn.DenseNet3(args.layers, 66, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #   10个label
        target = np.array(target).astype(int)
        target = torch.from_numpy(target)

        target = target.cuda(non_blocking=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = np.array(target).astype(int)
        target = torch.from_numpy(target)
        target = target.cuda(non_blocking=True)

        # print("input:", input)
        # print("target:", target)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_siam.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best_siam.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

     # print("maxk:", maxk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    # print("准确度：", res)
    return res


if __name__ == '__main__':
    main()