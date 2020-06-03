from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from networks import WaveNet
from dataset import AudioDataset

gpu_id = 0

net = WaveNet()
net = net.to(gpu_id)

# Generate samples
state_dict = torch.load(Path.cwd() / 'checkpoints' / '035.pth')
net.load_state_dict(state_dict['model'], strict=False)

samples = net.sample(1, 1000, device=gpu_id)
#plt.plot(samples[0, 0])
#plt.savefig('test.png')
np.save('samples.npy', samples)

'''
with torch.no_grad():
    x = torch.randn(1,1,8000).to(gpu_id)
    pred0 = net(x)
    x[0,0,50:650] = 129.32
    pred1 = net(x)
    np.save('test.npy', (pred0==pred1).cpu().detach().numpy())
'''
