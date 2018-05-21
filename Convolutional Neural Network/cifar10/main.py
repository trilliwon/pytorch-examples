import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms


from utils import get_progress_bar, update_progress_bar

# Module
import GoogLeNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Traning')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes (default=1)')
parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

args.distributed = args.world_size > 1
args.use_cuda = torch.cuda.is_available()

best_accuracy = 0
start_epoch = 0

# Data
print('===> Preparing Data')
transform_tran = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_tran)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume:
    print('===> Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['ephoc']
else:
    print('===> Building model...')
    net = GoogLeNet.GoogLeNet()

if args.distributed:
    dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size)

if args.distributed:
    if args.use_cuda:
        net = torch.nn.parallel.DistributedDataParallel(net)
        net.cuda()
    else:
        net = torch.nn.parallel.DistributedDataParallelCPU(net)
else:
    if args.use_cuda:
        net = torch.nn.parallel.DataParallel(net)
        net.cuda()

criterion = nn.CrossEntropyLoss().cuda() if args.use_cuda else nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

# Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    print('Train')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global best_accuracy
    progress_bar_obj = get_progress_bar(len(testloader))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        update_progress_bar(progress_bar_obj, index=batch_idx, loss=(train_loss / (batch_idx + 1)),
                            acc=(correct / total), c=correct, t=total)
        
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_accuracy:
            print('\nSaving..')
            state = {
                'net': net.module if args.use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_accuracy = acc

def test(epoch):
    print('\nTest')

    global best_accuracy
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    progress_bar_obj = get_progress_bar(len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        update_progress_bar(progress_bar_obj, index=batch_idx, loss=(test_loss / (batch_idx + 1)),
                            acc=(correct / total), c=correct, t=total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_accuracy:
        print('\nSaving..')
        state = {
            'net': net.module if args.use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_accuracy = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
