#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from haojiepaint import *


from toposim import Cluster, alltoall, allreduce
from predict import create_cluster, predict_all2all, predict_computation


cluster_nico = create_cluster(16)
cluster_th = Cluster()
cluster_th.create_nvlink_cluster(16)


def get_predictions(prefix, layer, iteration, count, d_model):
    ranks = list(range(16))
    cluster = cluster_nico
    filename = f'cache/{prefix}-l{layer:02d}-it{iteration}.pkl'
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
   


iters = [80500]
dms = [1024, 4096]

def get_estm(prefix, dm, iteration, layer):
    with open(f'logs/sims/{dm}/{prefix}-l{layer}-it{iteration}.pkl', 'r') as f:
        while True:
            g = eval(f.readline())
            t = eval(f.readline())
            if len(g['moe']) == 1:
                return t

def get_time(prefix, dm, layers=-1):
    tot_time = 0.
    no_moe = False
    if layers != -1:
        pass
    elif prefix.find('gpt') == -1 and prefix.find('tianhe') == -1:
        layers = 24
    else:
        layers = 12
    if not isinstance(layers, list):
        la = range(layers)
    else:
        la = layers
        layers = len(layers)
    results = []
    deviation = []
    for layer in la:
        layer_time = 0
        last_iter = 0
        timesave = np.zeros(4)
        for i in iters:
            filename = 'logs/{}/times-fastmoe/dm{}-l{}-it{}.pkl'.format(prefix, dm, layer, i)
            with open(filename, 'rb') as f:
                t = pickle.load(f)
            tbl = np.array(t).sum(axis=1)
            
            filename = 'logs/{}/times-smartsch/dm{}-l{}-it{}.pkl'.format(prefix, dm, layer, i)
            with open(filename, 'rb') as f:
                t = pickle.load(f)
            tov = np.array(t).sum(axis=1)
            
            tes = np.array(get_predictions(prefix, layer, i, len(tov), dm))
            tbl = tbl[2:]
            tov = tov[2:]
            tes = tes[2:]
            results.append((tbl, tov, tes))
    return results # [np.concatenate(s) for s in zip(*results)]


ys = []

for s in ['moe-gpt', 'moe-bert']:
    dms = [1024, 4096]
    for dm in dms:
        d = get_time(s, dm)
        y1, y2 = [], []
        for tbl, tov, tes in d:
            tcomm, tcomp = tes.transpose(1, 0)
            tth = (tcomm * 4 + tcomp * 3) / (np.maximum(tcomm * 2, tcomp) + np.maximum(tcomm * 2, tcomp * 2))
            y1.append(tth.mean())
            y2.append((tbl / tov).mean())
        ys.append((y1, y2))


# In[8]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 2)
wid = .9

thmax = []
rlmax = []
for i, (y1, y2) in enumerate(ys):
    xs = np.arange(len(y1)) * wid / len(y1) - wid / 2 + i
    w = wid / len(y1)
    if i > 0:
        mask = None
    else:
        mask = True
    ax.bar(xs, y1, width=w, color=color_def[1], edgecolor='black', label=mask and 'Theoretical')
    ax.bar(xs, y2, width=w, color=color_def[4], edgecolor='black', hatch=hatch_def[0], label=mask and 'Achieved')
    y1 = np.array(y1)
    y2 = np.array(y2)
    thmax.append(np.mean(y1))
    rlmax.append(np.mean(y2))

ax.plot([-1, len(ys)], [1, 1], color='gray', linestyle='--', linewidth=1)

ax.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

ax.set_xlim(-1 +  wid / 2, len(ys) - wid / 2)
ax.set_xticks(np.arange(len(ys)))
ax.set_xticklabels(['GPT-S', 'GPT-L', 'BERT-Deep', 'BERT-Deep-L'])

x = 4 - 1 / 2 - wid / 30
ax.plot([x, x], [0, 2], color='black', linestyle='--', linewidth=1)

ax.set_ylabel('Speedup')
ax.set_ylim(0, 2.0)

plt.savefig('results/fig12.pdf', bbox_inches='tight')

