from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from networks import WaveNet
from dataset import AudioDataset, DummyDataset

from tensorboardX import SummaryWriter

writer = SummaryWriter()

gpu_id = 4

net = WaveNet()
net = torch.nn.DataParallel(net, device_ids=[4,7])
net = net.to(gpu_id)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

#trainloader = DataLoader(AudioDataset('./data/'), 
trainloader = DataLoader(DummyDataset(), 
                         batch_size=170, 
                         shuffle=True,
                         num_workers=8)

cnt = 0
for epoch in range(50):
    for i, (data, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        data = data.float().to(gpu_id)
        target = target.to(gpu_id)

        pred = net(data)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss', loss.item(), cnt)
        cnt += 1

    torch.save({'model': net.state_dict(),
                'optimizer': optimizer.state_dict()},
                Path.cwd() / 'checkpoints' / f'{epoch:03d}.pth')
