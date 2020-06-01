import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from networks import WaveNet
from dataset import AudioDataset

net = WaveNet().cuda()
optimizer = optim.Adam(net.parameters(), lr=3e-3)

x = torch.randn(1,1,2000)

trainloader = DataLoader(AudioDataset('./data/denero/npy'), 
                         batch_size=16, 
                         shuffle=True)

for i, (data, target) in enumerate(trainloader):
    data = data.float().cuda()
    target = target.cuda()
    pred = net(data)

    loss = F.cross_entropy(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

