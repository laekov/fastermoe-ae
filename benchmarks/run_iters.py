import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from fmoe import FMoETransformerMLP
from fmoe.functions import count_by_gate
from gates import TraceRepGate
from predict import predict_mlp
import time
import pickle


rank = None
world_size = None
dev_name_default = "cuda:0"


def run_trace(rank, world_size, d_model, trace_args):
    n_iters = 64

    trace_path, trace_layer, trace_iter = trace_args
    gate = TraceRepGate(trace_path, trace_layer, [rank], world_size, trace_iter)
    batch_size = gate.batch_size

    gen_rep_gate = lambda d_m, n_e, w_s, topk: gate

    model = FMoETransformerMLP(
        num_expert=1,
        d_model=d_model,
        d_hidden=d_model * 2,
        world_size=world_size,
        activation=nn.ReLU(),
        gate=gen_rep_gate,
        top_k=2).cuda()

    model.train()

    times = []
    for i in range(n_iters):
        x = torch.rand(batch_size, d_model, device='cuda')
        x.requires_grad = True
        dist.barrier()
        
        tfs = time.time()
        y = model(x)
        dist.barrier()
        tfe = time.time()

        loss = y.sum()

        tbs = time.time()
        loss.backward()
        dist.barrier()
        tbe = time.time()

        model.gate.next()

        if rank == 0:
            tfw = (tfe - tfs) * 1e3
            tbw = (tbe - tbs) * 1e3
            print('It {} fw {} bw {} tot {}'.format(i, tfw, tbw, tfw + tbw))
            times.append((tfw, tbw))
    if rank == 0:
        pkl_path = 'logs/{}/times-{}/dm{}-l{}-it{}.pkl'.format(
                trace_path.split('/')[-1], os.environ['TEST_NAME'],
                d_model, trace_layer, trace_iter)
        with open(pkl_path, 'wb') as f:
            pickle.dump(times, f)

 
if __name__ == '__main__':
    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    d_model = int(sys.argv[1])
    trace_path = sys.argv[2]
    trace_layer = int(sys.argv[3])
    trace_iter = int(sys.argv[4])

    for tl in range(trace_layer):
        if rank == 0:
            print('    Iteration {} Layer {} / {}'.format(trace_iter, tl, trace_layer))
        trace_args = (trace_path, tl, trace_iter)
        run_trace(rank, world_size, d_model, trace_args)
