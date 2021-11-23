import os
import torch
import torch.distributed as dist
import torch.nn as nn
from fmoe.functions import prepare_forward
from fmoe.functions import MOEScatter, MOEGather, ensure_comm
from gates import TraceRepGate, RandomGate
import timer
from linear import Expert
from predict import predict_mlp


rank = None
world_size = None
dev_name_default = "cuda:0"


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size):
    n_runs = 16
    dist.barrier()
    timer.start('prepare')
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        _,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    dist.barrier()
    timer.stop('prepare')

    lat_comm, lat_mm = predict_mlp(local_expert_count, d_model, world_size)

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    for _ in range(n_runs):
        dist.barrier()
        timer.start('scatter')
        x = MOEScatter.apply(
                inp, pos // topk,
                local_expert_count, global_expert_count, 
                torch.zeros(world_size, dtype=torch.bool),
                fwd_batch_size, world_size
        )
        torch.distributed.barrier()
        timer.stop('scatter')

    if rank == 0:
        print('Scatter estm {:.4} ms real {:.4f} ms'.format(
            lat_comm * 1e3,
            timer.get('scatter')[0] * 1e3))

    for _ in range(n_runs):
        timer.start('expert')
        z = expert_fn(x, fwd_expert_count)
        dist.barrier()
        timer.stop('expert')

    t, _ = timer.get('expert')
    dist.barrier()

    if rank == 0:
        print('Computation estm {:.4} ms real {:.4f} ms'.format(
            lat_mm * 1e3,
            timer.get('expert')[0] * 1e3))

    out_batch_size = inp.shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    x = z

    timer.start('gather')
    x = MOEGather.apply(
            x, pos,
            local_expert_count, global_expert_count,
            torch.zeros(world_size, dtype=torch.bool),
            out_batch_size, world_size
    )
    torch.distributed.barrier()
    timer.stop('gather')
    return x



def run_trace(rank, world_size, d_model, trace_args):
    n_runs = 64

    trace_path, trace_layer, trace_iter = trace_args
    if trace_path.startswith('rand'):
        batch_size = int(trace_path[4:])
        gate = RandomGate(d_model, 1, world_size, 2, -3).cuda()
    else:
        tr_gate = TraceRepGate(trace_path, trace_layer, rank, world_size, trace_iter).cuda()
        gate = tr_gate
        batch_size = tr_gate.batch_size

    expert = Expert(d_model, d_model * 2).cuda()

    gen_rep_gate = lambda d_m, n_e, w_s, topk: gate

    for i in range(n_runs):
        x = torch.rand(batch_size, d_model).cuda()
        ensure_comm(x, None)

        x.requires_grad = True

        gate_idx, gate_score = gate(x)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        timer.start('forward')
        y = _fmoe_general_global_forward(x, gate_idx, lambda x, _: expert(x),
                1, world_size)
        timer.stop('forward')

        loss = y.sum()

        timer.start('backward')
        loss.backward()
        timer.stop('backward')

        if rank == 0:
            print('Iteration: {} | {}'.format(
                i, timer.report(['forward', 'backward'], window=1)))
            # print(timer.report(['prepare', 'scatter', 'expert', 'gather'], window=1))

 
if __name__ == '__main__':
    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    d_model = int(os.environ.get("D_MODEL", "1024"))
    trace_paths = os.environ.get('TRACE_PATH', '/home/laekov/Megatron-2.2/dump/gshard-gpt-1')
    for trace_path in trace_paths.split(','):
        for trace_layer in os.environ.get('TRACE_LAYER', '5').split(','):
            for trace_iter in os.environ.get('TRACE_ITER', '40500').split(','):
                trac_iter = int(trace_iter)
                trace_layer = int(trace_layer)
                trace_args = (trace_path, trace_layer, trace_iter)
                
                if rank == 0:
                    print('===')
                    print('Running trace {} layer {} iteration {}'.format(*trace_args))

                run_trace(rank, world_size, d_model, trace_args)
            if trace_path.startswith('rand'):
                break
