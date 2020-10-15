# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:58:24 2020

@author: Yun, Junhyuk
"""

from __future__ import print_function
import torch
from torch import nn, optim, cuda
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import numpy as np

# GPU settings
device = 'cuda' if cuda.is_available() else 'cpu'
print("Device: ", device, "\n");

# Training settings
batch_size = 60

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

latent_space_size=100

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.layer1 = nn.Sequential(
            nn.Linear(latent_space_size + 10, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels): # 200
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.layer1 = nn.Sequential(
            nn.Linear(28*28 + 10, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        x = x.view(-1, 28*28)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

G = Generator().to(device)
D = Discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=0.0001)
D_optimizer = optim.Adam(D.parameters(), lr=0.0001)
criterion = nn.BCELoss()

epochs=101
G_loss_arr=[]
D_loss_arr=[]
for epoch in range(epochs):
    G_loss_avg=0
    D_loss_avg=0
    for batch_idx, (real_data, real_label) in enumerate(train_loader):
        
        real_data = real_data.to(device)
        real_label = real_label.to(device)
          
        is_real = Variable(torch.ones(batch_size, 1, device=device))
        is_fake = Variable(torch.zeros(batch_size, 1, device=device))
        
        # Scenario1
        z = Variable(torch.randn(batch_size, latent_space_size, device=device))
        fake_label = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
        torch.LongTensor()
        fake_data = G(z, fake_label)
        
        D_result_from_fake = D(fake_data, fake_label)
        G_loss = criterion(D_result_from_fake, is_real)
        
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
        # Scenario2
        D_result_from_real = D(real_data, real_label)
        D_loss_real = criterion(D_result_from_real, is_real)
    
        z = Variable(torch.randn(batch_size, latent_space_size, device=device))
        fake_label = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
        fake_data = G(z, fake_label)
        D_result_from_fake = D(fake_data, fake_label)
        D_loss_fake = criterion(D_result_from_fake, is_fake)

        D_loss = D_loss_real + D_loss_fake
        
        
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        G_loss_avg += G_loss
        D_loss_avg += D_loss
        
        if batch_idx % 200 == 0:
            print('Epoch: {} [{}/{}({:.0f}%)] G_loss: {:.2f} D_loss: {:.2f}'.format(
                epoch+1, batch_idx * len(real_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), G_loss.item(), D_loss.item()))

        
    G_loss_avg /= batch_size
    D_loss_avg /= batch_size
    G_loss_arr.append(G_loss)
    D_loss_arr.append(D_loss)

    if (epoch+1) % 10 == 1 or epoch+1<=10:
        print('Epoch: {}\tG_loss: {:.2f} D_loss: {:.2f}'.format(epoch+1, G_loss_avg, D_loss_avg))

        z = Variable(torch.randn(batch_size, latent_space_size, device=device))
        fake_label = Variable(torch.LongTensor(np.zeros(batch_size))).to(device)
        for i in range(len(fake_label)):
            fake_label[i]=i%10
        fake_data = G(z, fake_label)
        
        fake_data = fake_data.reshape([batch_size, 1, 28, 28])
        img_grid = make_grid(fake_data, nrow=10, normalize=True)
        save_image(img_grid, "CGAN-fake-img/epoch%03d.png"%(epoch+1))

from matplotlib import pyplot as plt

plt.plot([i+1 for i in range(len(G_loss_arr))], G_loss_arr)
plt.plot([i+1 for i in range(len(D_loss_arr))], D_loss_arr)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('CAN')
plt.legend(['G_loss', 'D_loss'])
plt.show()
