import os
import torch
import torch.distributed as dist
import torch.nn as nn
from fmoe import FMoETransformerMLP
from fmoe.functions import count_by_gate
from gates import TraceRepGate
from predict import predict_mlp
import timer


rank = None
world_size = None
dev_name_default = "cuda:0"


def run_trace(rank, world_size, d_model, trace_args):
    n_runs = 4
    n_iters = 10

    trace_path, trace_layer, trace_iter = trace_args
    gate = TraceRepGate(trace_path, trace_layer, rank, world_size, trace_iter)
    batch_size = gate.batch_size

    gen_rep_gate = lambda d_m, n_e, w_s, topk: gate

    model = FMoETransformerMLP(
        num_expert=1,
        d_model=d_model,
        d_hidden=d_model * 2,
        world_size=world_size,
        gate=gen_rep_gate,
        top_k=2).cuda()

    model.train()

    for i in range(n_iters):
        timer.clear()
        for _ in range(n_runs):
            x = torch.rand(batch_size, d_model, device='cuda')
            x.requires_grad = True
            dist.barrier()
            
            timer.start('forward')
            y = model(x)
            dist.barrier()
            timer.stop('forward')

            loss = y.sum()

            timer.start('backward')
            loss.backward()
            dist.barrier()
            timer.stop('backward')

        gate_idx = model.gate.current()
        _, lec, gec = count_by_gate(gate_idx, 1, world_size, require_pos=False)
        lat_comm, lat_mm = predict_mlp(lec, d_model, world_size)
        lat_fw = lat_comm * 2 + lat_mm
        lat_bw = lat_comm * 2 + lat_mm * 2

        model.gate.next()

        if rank == 0:
            print('Prediction fw {:.3f} ms bw {:.3f} ms'
                  'Real fw {:.3f} ms bw {:.3f} ms'.format(
                        lat_fw * 1e3, lat_bw * 1e3,
                        timer.get('forward', 2)[0] * 1e3,
                        timer.get('backward', 2)[0] * 1e3))

 
if __name__ == '__main__':
    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    d_models = os.environ.get("D_MODEL", "1024")
    trace_path = os.environ.get('TRACE_PATH')
    trace_layer = int(os.environ.get('TRACE_LAYER', '0'))
    trace_iter = int(os.environ.get('TRACE_ITER', '80500'))
    trace_args = (trace_path, trace_layer, trace_iter)

    for d_model in d_models.split(','):
        if rank == 0:
            print('D_MODEL {} Model {} layer {} it {}'.format(d_model, trace_path, trace_layer, trace_iter))
        run_trace(rank, world_size, int(d_model), trace_args)
