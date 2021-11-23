import os
import torch
import torch.distributed as dist
import torch.nn as nn
from fmoe import FMoETransformerMLP
from fmoe.functions import MOEScatter, MOEGather
import fmoe_cuda
from gates import AllGoGate, FlatGate, RandomGate
import timer


rank = None
world_size = None
dev_name_default = "cuda:0"


class Expert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def perf_linear(batch_size, d_model, d_hidden):
    model = Expert(d_model, d_hidden).cuda()

    n_runs = 0

    while True:
        x = torch.rand(batch_size, d_model, device=model.fc1.weight.device)
        x.requires_grad = True
        torch.cuda.synchronize()
        timer.start('forward')
        y = model(x)
        torch.cuda.synchronize()
        timer.stop('forward')

        z = y.sum()
        timer.start('backward')
        z.backward()
        torch.cuda.synchronize()
        timer.stop('backward')

        n_runs += 1
        if n_runs > 10 and n_runs * timer.get('forward')[0] > 1.:
            break
    m, v = timer.get('forward')
    fw_gf = batch_size * d_model * d_hidden / m * 4e-9
    m, v = timer.get('backward')
    bw_gf = batch_size * d_model * d_hidden / m * 8e-9
    print('{}x{}x{} | fw {:.4f} GFLOPs | bw {:.4f} GFLOPs | {}'.format(
        batch_size, d_model, d_hidden, fw_gf, bw_gf,
        timer.report(['forward', 'backward'])))

 
if __name__ == '__main__':
    for d_m in [1024, 4096]:
        for hidden_frac in [2, 4, 8, 16]:
            for batch_size in range(1, 4096):
                d_h = d_m * hidden_frac
                perf_linear(batch_size, d_m, d_h)
            for batch_size in range(10000, 20000, 1000):
                d_h = d_m * hidden_frac
                perf_linear(batch_size, d_m, d_h)
