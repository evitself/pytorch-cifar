'''Train CIFAR10 with PyTorch.'''
import imp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# auto mixed precision
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from scheduler.cosine_annealing_lr_with_warmup import CosineAnnealingWarmupRestarts


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--amp', '-m', action='store_true',
                    help='enable auto-mix-precision training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

scaler = GradScaler() if args.amp else None

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

TOTAL_EPOCH = 100

# Data
INPUT_DIM = 128
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((INPUT_DIM, INPUT_DIM)),
    transforms.RandomCrop(INPUT_DIM, padding=8),
    # transforms.RandomResizedCrop(INPUT_DIM, scale=(0.8, 1)),
    # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
    transforms.RandomAdjustSharpness(2, p=0.5),
    transforms.RandomApply([transforms.AugMix()], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.5, 1.5))], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    AddGaussianNoise(0, 1)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class_weights = torch.FloatTensor([0.285, 0.715])
if device == 'cuda':
    class_weights = class_weights.cuda()
# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, 32)

custom_data = torchvision.datasets.ImageFolder(
    root='./data/cars_9_data', transform=transform_train)
total_length = len(custom_data)
split_length = int(0.1 * total_length)

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
testset, trainset = torch.utils.data.random_split(custom_data, [split_length, total_length - split_length],
                                         generator=torch.Generator().manual_seed(99))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')
classes = tuple(sorted(custom_data.class_to_idx.items(), key=lambda _x: _x[0]))
num_classes = len(classes)

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA(num_classes=num_classes)
# net = PreActResNet18(num_classes=num_classes);
net = PreActResNet18(num_classes=num_classes);
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCH)
scheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, min_lr=args.lr*0.001,
                                          first_cycle_steps=50, cycle_mult=1.0,
                                          warmup_steps=10, gamma=0.2)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    cur_acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.amp:
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            loss = scaler.scale(loss)
            scaler.step(optimizer)            
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        cur_acc = correct / total

        cur_lr = scheduler.get_lr()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %f'
                     % (train_loss/(batch_idx+1), 100.*cur_acc, correct, total, cur_lr[0]))

        if args.amp:
            scaler.update()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+TOTAL_EPOCH):
    train(epoch)
    test(epoch)
    scheduler.step()
