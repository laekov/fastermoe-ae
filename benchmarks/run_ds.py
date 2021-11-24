import os
import pickle
import torch
import torch.distributed as dist
import torch.nn as nn
from fmoe import FMoETransformerMLP
from fmoe.functions import count_by_gate
from gates import TraceRepGate
from predict import predict_mlp
import deepspeed
import argparse
import time

rank = None
world_size = None
device_name = "cuda:0"


def run_trace(rank, world_size, d_models, trace_args):
    n_iters = 10

    trace_path, trace_layer, trace_iter = trace_args
    gate = TraceRepGate(trace_path, trace_layer, rank, 16, trace_iter)
    batch_size = gate.batch_size

    gen_rep_gate = lambda d_m, n_e, w_s, topk: gate

    for d_model in d_models:
        times = []
        mems = []
        if rank == 0:
            print('trace_path {} batch_size {} d_model {} trace_layer {} trace_iter {}'.format(
                trace_path, batch_size, d_model, trace_layer, trace_iter))
        model = FMoETransformerMLP(
                num_expert=16,
                d_model=d_model,
                d_hidden=d_model * 2,
                world_size=1,
                gate=gen_rep_gate,
                top_k=2).to(device_name)

        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = deepspeed.add_config_arguments(parser)
        args, _ = parser.parse_known_args()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=args,
                lr_scheduler=None,
                mpu=None,
                dist_init_required=False)

        model.train()
        # model.module.gate.reset()

        memb = torch.cuda.max_memory_allocated(device=None)/1024/1024
        if rank == 0:
            print('device: {} max memory allocated while building: {} MB'.format(rank, memb))
        torch.cuda.reset_peak_memory_stats(device=None)

        for i in range(n_iters):
            x = torch.rand(batch_size, d_model, device=device_name)
            x.requires_grad = True
            dist.barrier()

            tfs = time.time()
            y = model(x)
            dist.barrier()
            tfe = time.time()

            loss = y.sum()

            tbs = time.time()
            # loss.backward()
            model.backward(loss)
            dist.barrier()
            tbe = time.time()

            model.module.gate.next()
            if rank == 0:
                tfw = (tfe - tfs) * 1e3
                tbw = (tbe - tbs) * 1e3
                print('It {} fw {} bw {} tot {}'.format(i, tfw, tbw, tfw + tbw))
                times.append((tfw, tbw))

        memr = torch.cuda.max_memory_allocated(device=None)/1024/1024
        if rank == 0:
            print('device: {} max memory allocated: {} MB'.format(rank, memr))
        torch.cuda.reset_peak_memory_stats(device=None)
        if rank == 0:
            mems.append((memb, memr))

        if rank == 0:
            stage = os.environ.get("DSP_STAGE")
            pkl_path = 'logs/{}/times-ds-{}/bsz{}-dm{}-l{}-it{}.pkl'.format(
                trace_path.split('/')[-1], stage,
                batch_size * 2, d_model, trace_layer, trace_iter)
            with open(pkl_path, 'wb') as f:
                pickle.dump((times, mems), f)
 
if __name__ == '__main__':
    if int(os.environ["WORLD_SIZE"]) > 1:
        if 'OMPI_COMM_WORLD_RANK' in os.environ:
            os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        deepspeed.init_distributed(auto_mpi_discovery=False)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
        deepspeed.init_distributed(auto_mpi_discovery=False)
    d_models = os.environ.get("D_MODEL")
    d_models = list(map(int, d_models.split(',')))
    trace_path = os.environ.get('TRACE_PATH')
    trace_layer = int(os.environ.get('TRACE_LAYER', '0'))
    trace_iter = int(os.environ.get('TRACE_ITER', '80500'))
    trace_args = (trace_path, trace_layer, trace_iter)
      
    run_trace(rank, world_size, d_models, trace_args)
