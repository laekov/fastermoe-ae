import pickle
from toposim import Cluster, alltoall, allreduce
from predict import create_cluster, predict_all2all, predict_computation


cluster_nico = create_cluster(16)
cluster_th = Cluster()
cluster_th.create_nvlink_cluster(16)


def get_predictions(prefix, layer, iteration, count, d_model):
    if prefix == 'tianhe':
        ranks = list(range(64))
        cluster = cluster_th
        filename = f'/home/laekov/fastmoe-perf/cache/{prefix}/l{layer}-it{iteration}.pkl'
    else:
        ranks = list(range(16))
        cluster = cluster_nico
        filename = f'/mnt/zoltan/laekov/dump/cache/{prefix}-l{layer}-it{iteration}.pkl'
    with open(filename, 'rb') as f:
        flow_mats = pickle.load(f)
    res = []
    for i in range(count):
        flow_mat = flow_mats[i]
        cluster.reset_traffic()
        alltoall(cluster, ranks, flow_mat, dsize=4 * d_model)
        lat_moe = cluster.get_latency()
        lat_comp = predict_computation(flow_mat, d_model, 2)
        res.append((lat_moe, lat_comp))
    return res
    
