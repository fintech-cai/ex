import torch
x = torch.load('train.pt')
print(x['samples'].shape)