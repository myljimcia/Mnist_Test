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
'''
x = torch.rand(4,2,3,3)
print(x)

y = x.view(-1,1,3,3)
#print(y)

min = torch.min(x)
max = torch.max(x)
mean = torch.mean(x)
median = torch.median(x)
std = torch.std(x)

sign = torch.sign(x)

print(x[sign == -1])
print(x.view(-1).size(0))
'''
a = torch.rand(10,3,4,4)
b = torch.rand(10,2,3,3)
tuple = (a,b)
tuple2 = ('a','b')

b=21

a = ['a',b,'c']

print(a)



