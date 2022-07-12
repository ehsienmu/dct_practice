import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import torch_dct as dct
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import adversarial_attacks_pytorch_master.torchattacks as torchattacks
from adversarial_attacks_pytorch_master.torchattacks import PGD, FGSM

from tqdm.auto import tqdm
# from models import CNN

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

class mydct(object):
    def __call__(self, x):
        return dct.dct(x)  #Image.fromarray(t)

    def __repr__(self):
        return self.__class__.__name__+'()'


train_tfm_dct = transforms.Compose([
    transforms.ToTensor(),
    mydct(),
])
test_tfm_dct = transforms.Compose([
    transforms.ToTensor(),
    mydct(),
])

train_tfm = transforms.Compose([
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([
    transforms.ToTensor(),
])

# def prepare_loaders(BATCH_SIZE=128, dataset='mnist', do_dct=False):
#     if dataset == 'mnist':
#         DSET = datasets.MNIST
#     elif dataset == 'cifar10':
#         DSET = datasets.CIFAR10
#     else:
#         print('not found')

#     if(do_dct):
#         train_dataset = DSET(root='~/DATA', train=True, transform=train_tfm_dct, download=True)
#         test_dataset = DSET(root='~/DATA', train=False, transform=test_tfm_dct, download=True)
#     else:
#         train_dataset = DSET(root='~/DATA', train=True, transform=train_tfm, download=True)
#         test_dataset = DSET(root='~/DATA', train=False, transform=test_tfm, download=True)


#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     return train_loader, test_loader

mnist_train = dsets.MNIST(root='~/DATA',
                          train=True,
                          transform=train_tfm_dct,
                          download=True)

mnist_test = dsets.MNIST(root='~/DATA',
                         train=False,
                         transform=test_tfm_dct,
                         download=True)

batch_size = 128

train_loader  = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=False)


model = models.resnet18()
model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.fc = torch.nn.Linear(512, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
atk = PGD(model, eps=0.4, alpha=0.1, steps=7)
num_epochs = 5

correct = 0
total = 0
for epoch in range(num_epochs):

    total_batch = len(mnist_train) // batch_size
    
    pbar = tqdm(train_loader, ncols=80, desc='train epoch: '+str(epoch), position=0,leave=True)
    for  batch_images, batch_labels in pbar:#enumerate(train_loader):
        X = atk(batch_images, batch_labels).cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        _, predicted = torch.max(pre.data, 1)

        total += Y.size(0)
        correct += (predicted ==Y).sum()
        acc = (100 * float(correct) / total)
        pbar.set_postfix(loss=cost.item(), accu=acc)
        
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    images = images.cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
model.eval()

correct = 0
total = 0

# atk = FGSM(model, eps=0.3)

for images, labels in test_loader:
    
    images = atk(images, labels).cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
