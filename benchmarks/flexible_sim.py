import os
import sys
import pickle
import numpy as np

from toposim import Cluster, alltoall, allreduce
from predict import create_cluster, predict_all2all, predict_computation
from gen_group import gen_groups, enumerate_groups


def process_lec(filename, world_size, data=None):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    n_experts = len(data[0])
    if n_experts == world_size:
        return data
    epw = n_experts // world_size
    for i in range(len(data)):
        data[i] = np.array(data[i]).reshape(world_size, epw, world_size, epw)
        data[i] = data[i].sum(axis=1).sum(axis=2)
    return data


def sim_trace(cluster, flow_mat, groups, d_model, hidden_factor, world_size):
    batch_size = np.array(flow_mat).sum() // (len(groups['moe'][0]) * len(groups['dp'][0]))
    lat_comp = predict_computation(flow_mat, d_model, hidden_factor)

    cluster.reset_traffic()
    for g in groups['moe']:
        alltoall(cluster, g, flow_mat, dsize=4 * d_model)
    lat_moe = cluster.get_latency()

    cluster.reset_traffic()
    for g in groups['mp']:
        if len(g) > 1:
            allreduce(cluster, g, batch_size * d_model)
    lat_mp = cluster.get_latency()

    cluster.reset_traffic()
    for g in groups['dp']:
        if len(g) > 1:
            allreduce(cluster, g, 2 * hidden_factor * (d_model**2))
    lat_dp = cluster.get_latency()

    lat_e2e = lat_comp * 3 + lat_moe * 4 + lat_mp * 2 + lat_dp

    return np.array([lat_comp, lat_moe, lat_mp, lat_dp, lat_e2e]) * 1e3


if __name__ == '__main__':
    filename = sys.argv[1]
    d_model = int(os.environ.get("D_MODEL", "1024"))
    world_size = int(os.environ.get("WORLD_SIZE", "16"))
    cluster = create_cluster(world_size)
    gdata = []
    gmoe, gdp, gmp = gen_groups(world_size, 1, 1)
    groups = dict(moe=gmoe, dp=gdp, mp=gmp)
    flow_mats = process_lec(filename, len(groups['moe'][0]))

    res = []
    for m in flow_mats:
        res.append(sim_trace(cluster, m, groups, d_model,
                4 / len(groups['mp'][0]), world_size))
    res = np.array(res)
    gdata.append((groups, res.mean(axis=0).tolist()))

    if "LOG_PREFIX" not in os.environ:
        outst = sys.stdout
    else:
        outst = open(os.path.join(os.environ['LOG_PREFIX'], filename.split('/')[-1]), 'w')
    print('Predicted {} / {}'.format(filename, d_model))
    for i, (grp, res) in enumerate(gdata):
        outst.write(str(grp).replace(' ', '') + '\n')
        outst.write(str(res) + '\n')
        if outst == sys.stdout and i > 5:
            break
