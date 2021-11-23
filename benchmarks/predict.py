import numpy as np
from toposim import Cluster, alltoall


def allgather_data(local_expert_count, world_size):
    import torch
    import torch.distributed as dist
    l = local_expert_count.cuda()
    rcv = [torch.empty_like(l) for _ in range(world_size)]
    dist.all_gather(rcv, l)
    return [l.cpu().numpy() for l in rcv]


def create_cluster(world_size):
    cluster = Cluster()
    bw_pcie = 88 * 1e9 / 8
    bw_net = 50 * 1e9 / 8
    cluster.create_pcie_gpu_cluster(world_size // 8, 2, 4,
            bw_net=bw_net, bw_pcie=bw_pcie)
    return cluster


def predict_all2all(cluster, flow_mat, ranks, d_model):
    alltoall(cluster, ranks, flow_mat, 4 * d_model)
    lat_comm = cluster.get_latency()
    return lat_comm


def predict_computation(flow_mat, d_model, hidden_factor, bw_mm=11.5e12):
    fm = np.array(flow_mat)
    max_bs = fm.sum(axis=0).max()
    amt_mm = 2 * 2 * (hidden_factor * d_model) * d_model * max_bs
    lat_mm = amt_mm / bw_mm
    return lat_mm


def predict_latency(flow_mat, d_model, world_size, hidden_factor=2):
    cluster = create_cluster(world_size)
    lat_comm = predict_all2all(cluster, flow_mat, list(range(world_size)), d_model)
    lat_mm = predict_computation(flow_mat, d_model, hidden_factor)

    amt_fw = 2 * 2 * (hidden_factor * d_model) * d_model * (np.array(flow_mat).sum() / world_size)
    # amt_fw = 0

    return lat_comm, lat_mm, amt_fw


def predict_mlp(local_expert_count, d_model, world_size):
    flow_mat = allgather_data(local_expert_count, world_size)
    lat_comm, lat_mm, amt_fw = predict_latency(flow_mat, d_model, world_size)
    return lat_comm, lat_mm
