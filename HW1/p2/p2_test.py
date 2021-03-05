# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:56:13 2020

@author: Yayati
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module): 
      def __init__(self):
          super(Net, self).__init__()
          # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
          self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
          self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
          self.bn1=nn.BatchNorm2d(64)

          self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
          self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
          self.bn2=nn.BatchNorm2d(128)

          self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
          self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
          self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
          self.bn3=nn.BatchNorm2d(256)

          self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
          self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn4=nn.BatchNorm2d(512)

          self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
          self.bn5=nn.BatchNorm2d(512)

          
          self.pool = nn.MaxPool2d(2, 2)

          
          self.fc6 = nn.Linear(1*1*512, 4096)
          
        
          self.fc7 = nn.Linear(4096, 4096)
          
          
          self.fc8 = nn.Linear(4096, 1000)

      def forward(self, x, training=True):
        
          x = F.relu(self.conv1_1(x))
          x = F.relu(self.bn1(self.conv1_2(x)))
          x = self.pool(x)
        
          x = F.relu(self.conv2_1(x))
          x = F.relu(self.bn2(self.conv2_2(x)))
          x = self.pool(x)
          x = F.dropout(x, 0.2)
          #batchnorm

          x = F.relu(self.conv3_1(x))
          x = F.relu(self.conv3_2(x))
          x = F.relu(self.bn3(self.conv3_3(x)))
          x = self.pool(x)
          x = F.dropout(x, 0.2)

          x = F.relu(self.conv4_1(x))
          x = F.relu(self.conv4_2(x))
          x = F.relu(self.bn4(self.conv4_3(x)))
          x = self.pool(x)
          x = F.dropout(x, 0.2)

          x = F.relu(self.conv5_1(x))
          x = F.relu(self.conv5_2(x))
          x = F.relu(self.bn5(self.conv5_3(x)))
          x = self.pool(x)
          
          x = x.view(-1, 1 * 1 * 512)
          
          x = F.relu(self.fc6(x))
          # x = F.dropout(x, 0.2)

          

          x = F.relu(self.fc7(x))
          # x = F.dropout(x, 0.2)


          x = F.log_softmax(self.fc8(x))
          return x
net=Net()
print(net)

net = Net().to(device)
net.load_state_dict(torch.load('p2_model.pkl'))
net.eval()
dataiter = iter(testloader)
images, labels = dataiter.next()
images=images.to(device)
labels=labels.to(device)
outputs = net(images)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images=images.to(device)
        labels=labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
