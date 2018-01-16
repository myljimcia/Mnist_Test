from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import datasets,transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import argparse
import cv2
import time
import os
import math
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch MNIST Test')

parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--use_norm', action='store_false', default='store_true',
                    help='use norm before training')
parser.add_argument('--pooling_method', choices=['max','avg','stoch','stride_conv'],default='stride_conv',
                    help='pooling_method choosing from max avg stoch stride_conv')

args = parser.parse_args()

#config = 'PL_'+'stride_conv avg'+'_BZ_'+str(args.batch_size)+'_LR_'+str(args.lr)+'_Momentum_'+ str(args.momentum)
config = 'Weights Init_'+'xavier_normal_'+ 'PL_'+args.pooling_method+'_BZ_'+str(args.batch_size)+'_LR_'+str(args.lr)+'_Momentum_'+ str(args.momentum)

use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if args.use_norm:
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
else:
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

trainset = datasets.MNIST(root='./data',train=True,download=True,transform=transforms)
testset = datasets.MNIST(root='./data',train=False,download=True,transform=transforms)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,**kwargs)
testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=True,**kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1,8,3,padding=1,bias=False)

        self.bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,32,3,padding=1,bias=False)

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,padding=1,bias=False)

        if args.pooling_method == 'stride_conv':
            self.pl2 = nn.Conv2d(32,32,kernel_size=2,stride=2)
            self.pl3 = nn.Conv2d(64,64,kernel_size=14,stride=14)
        elif args.pooling_method == 'max':
            self.pl2 = nn.MaxPool2d(2)
            self.pl3 = nn.MaxPool2d(14)
        elif args.pooling_method == 'avg':
            self.pl2 = nn.AvgPool2d(2)
            self.pl3 = nn.AvgPool2d(14)
        elif args.pooling_method == 'stoch':
            self.pl2 = nn.FractionalMaxPool2d(2,output_size=14)
            self.pl3 = nn.FractionalMaxPool2d(14,output_size=1)


        #self.pl2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        #self.pl3 = nn.MaxPool2d(14)
        self.fc = nn.Linear(64,10)


    def forward(self,x):
        out1 = self.conv1(x)
        out2_bn = self.bn2(out1)
        out2_relu = F.relu(out2_bn)
        out2_conv = self.conv2(out2_relu)
        out2_pl = self.pl2(out2_conv)

        out3_conv = self.conv3(F.relu(self.bn3(out2_pl)))
        out3_pl = self.pl3(out3_conv)
        out = out3_pl.view(out3_pl.size(0),-1)
        out = self.fc(out)

        return out,(x,out1,out2_bn,out2_relu,out2_conv,out2_pl,out3_conv,out3_pl)

net = Net()

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
        #print(m.weight.size())
        nn.init.xavier_normal(m.weight,gain = math.sqrt(2))
        #nn.init.xavier_uniform(m.weight, gain=math.sqrt(2))
        #nn.init.kaiming_normal(m.weight, mode = 'fan_out' )
        #nn.init.kaiming_uniform(m.weight, mode = 'fan_out')


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)

a = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
writer = SummaryWriter('./log/'+'Mnist_'+config+'/')

def train(epoch):
    print('\nEpoch: %d'% epoch)
    print('\nTraining..: \n')

    net.train()

    cur_time = time.time()

    train_loss = 0
    correct = 0
    total = 0

    error_list = []

    del error_list[:]

    for batch_idx, (inputs,targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(),targets.cuda()
        inputs,targets = Variable(inputs),Variable(targets)
        optimizer.zero_grad()
        outputs,images =  net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _,predicted = torch.max(outputs.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        for ind, i in enumerate(predicted):
            if predicted[ind] != targets.data[ind]:
                error_list.append(inputs.data[ind,:,:,:])


        step_ratio = epoch +  (batch_idx+1) / len(trainloader)

        step = int(step_ratio * len(trainloader))

        step_ratio = round(step_ratio,2)

        #print('step ',step)
        #print('batch_idx ',batch_idx)
        #print('len(trainloader) ',len(trainloader))
        loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total

        if (batch_idx+1) == len(trainloader) :
            i = 1

            for name, param in net.named_parameters():
                param_tensor = param.clone().cpu().data

                writer.add_histogram(name, param_tensor.clone().numpy(),epoch)
                #print(str(name),param_tensor.size())
                if param_tensor.dim() == 4 :
                    if param_tensor.size(2) == 3:
                        param_tensor_3 = param_tensor.clone().view(-1,1,3,3)
                        x = vutils.make_grid(param_tensor_3,normalize=True,scale_each=True,padding=1)
                        #print(x.size())
                        writer.add_image('conv'+str(i), x, epoch)

                        sign = torch.sign(param_tensor)

                        pos = param_tensor[sign == 1]
                        neg = param_tensor[sign == -1]

                        min = torch.min(param_tensor)
                        max = torch.max(param_tensor)
                        mean = torch.mean(param_tensor)
                        median = torch.median(param_tensor)
                        std = torch.std(param_tensor)
                        sum = torch.sum(param_tensor)
                        count = param_tensor.view(-1).size(0)

                        pmin = torch.min(pos)
                        pmax = torch.max(pos)
                        pmean = torch.mean(pos)
                        pmedian = torch.median(pos)
                        pstd = torch.std(pos)
                        psum = torch.sum(pos)
                        pcount = pos.view(-1).size(0)

                        nmin = torch.min(neg)
                        nmax = torch.max(neg)
                        nmean = torch.mean(neg)
                        nmedian = torch.median(neg)
                        nstd = torch.std(neg)
                        nsum = torch.sum(neg)
                        ncount = neg.view(-1).size(0)

                        p_n_count = pcount/ncount
                        p_n_sum = abs(psum/nsum)

                        writer.add_scalar('conv' + str(i) + 'min', min, epoch)
                        writer.add_scalar('conv' + str(i) + 'max', max, epoch)
                        writer.add_scalar('conv' + str(i) + 'median', median, epoch)
                        writer.add_scalar('conv' + str(i) + 'std', std, epoch)
                        writer.add_scalar('conv' + str(i) + 'sum', sum, epoch)

                        writer.add_scalar('conv' + str(i) + 'p_n_count', p_n_count, epoch)
                        writer.add_scalar('conv' + str(i) + 'p_n_sum', p_n_sum, epoch)
                    elif param_tensor.size(2) == 2:
                        param_tensor_2 = param_tensor.clone().view(-1, 1, 2, 2)
                        x = vutils.make_grid(param_tensor_2, normalize=True, scale_each=True, padding=1)
                        # print(x.size())
                        writer.add_image('Stride_conv' + str(i), x, epoch)

                    i += 1

            #print(len(error_list))

            x = vutils.make_grid(error_list, normalize=True, scale_each=True)
            #print(x.size())
            writer.add_image('Error Trainning', x, epoch)

            image_names = ('x','out1','out2_bn','out2_relu','out2_conv','out2_pl','out3_conv','out3_pl')

            for a,b in zip(images,image_names):
                a = a.cpu().data
                a = a.view(-1, 1, a.size(2), a.size(3))
                if a.size(2) != 1:
                    x = vutils.make_grid(a, normalize=True, scale_each=True, padding=1)
                else:
                    x = vutils.make_grid(a, normalize=True, scale_each=False,range=(0,1), padding=1)
                writer.add_image('image_' + b, x, epoch)

        if (epoch+1) == args.epochs and  (batch_idx+1) == len(trainloader):
            print("final training error list is done")
            error_training = deepcopy(error_list)
            print(len(error_training))

        last_time = time.time()
        total_time = last_time - cur_time

        if (batch_idx+1) % 11.75 == 0 or batch_idx == 0 :
            print('Epoch: %.2f  |  Index: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | total time: %.3fs'
                  % (step_ratio ,(batch_idx + 1), train_loss / (batch_idx + 1), 100. * correct / total, correct,
                     total, total_time))

            writer.add_scalar('Training Loss', loss, step)
            writer.add_scalar('Training Acc', acc, step)



def test(epoch):

    print('\nTesting..: \n')

    net.eval()

    cur_time = time.time()

    test_loss = 0
    correct = 0
    total = 0

    error_list = []

    del error_list[:]

    for batch_idx, (inputs,targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(),targets.cuda()
        inputs,targets = Variable(inputs),Variable(targets)
        outputs,images =  net(inputs)
        loss = criterion(outputs,targets)

        test_loss += loss.data[0]
        _,predicted = torch.max(outputs.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        for ind, i in enumerate(predicted):
            if predicted[ind] != targets.data[ind]:
                error_list.append(inputs.data[ind,:,:,:])

        last_time = time.time()
        total_time = last_time - cur_time

        step_ratio = epoch +  (batch_idx+1) / len(testloader)

        step = int(step_ratio * len(testloader))

        step_ratio = round(step_ratio,2)

        #print('step ',step)
        #print('batch_idx ',batch_idx)
        #print('len(trainloader) ',len(trainloader))
        loss = test_loss / (batch_idx + 1)
        acc = 100. * correct / total


        if (batch_idx+1) % 5 == 0 or batch_idx == 0 :
            print('Epoch: %.2f  |  Index: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) | total time: %.3fs'
                  % (step ,(batch_idx + 1), test_loss / (batch_idx + 1), 100. * correct / total, correct,
                     total, total_time))

            writer.add_scalar('Testing Loss', loss, step)
            writer.add_scalar('Testing Acc', acc, step)

        if (batch_idx + 1) == len(testloader):
            x = vutils.make_grid(error_list, normalize=True, scale_each=True)
            print(x.size())
            writer.add_image('Error Testing', x, epoch)


        if (epoch+1) == args.epochs and  (batch_idx+1) == len(testloader):
            print("final test error list is done")
            error_test = deepcopy(error_list)
            print(len(error_test))


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

writer.close()











