#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pickle
import numpy as np
import torch
from haojiepaint import *
from utils import *


# In[157]:


# copied from fmoe impl
def policy_fn(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused):
    bw_pcie = 88 * 1e9 / 8
    bw_net = 50 * 1e9 / 8
    bw_mm = 11.5e12
    data_size = 4
    alpha = 2

    if fused:
        all_experts_count = all_experts_count.sum(dim=-1).view(world_size, world_size, 1)
        all_global_expert_count = all_global_expert_count.sum(dim=-1).view(world_size, world_size, 1)

    fwd_expert_counts = all_global_expert_count.sum(1) # [world_size, num_expert]
    default_counts = fwd_expert_counts.clone()

    _, indices = fwd_expert_counts.sort(0, descending=True)

    alphaHsquared = alpha * d_model ** 2 * data_size

    B_w = default_counts.max(0)[0]
    lat_comp = 3 * 4 * B_w * alphaHsquared / bw_mm  + 4 * B_w * d_model * data_size / bw_net

    comm = float('+inf')

    model_size = 2 * alphaHsquared * num_expert / bw_net * 2 / world_size
    comp_time = 12 * alphaHsquared / bw_mm

    for i, index in enumerate(indices):
        fwd_expert_counts[index] = 0
        fwd_expert_counts += all_global_expert_count[index].view(world_size, -1)

        B_k = fwd_expert_counts.max(0)[0]
        lat_comm = fwd_expert_counts.max(0)[0] * comp_time + (i+1) * model_size

        if lat_comm < comm:
            comm = lat_comm
        elif lat_comm > comm:
            break

    res = all_experts_count.new_zeros(world_size, num_expert, dtype=bool)

    if lat_comp > comm:
        res[indices[:i]] = True
    # print(lat_comp, comm)
    return res, lat_comp - comm


# In[154]:


def get_repl(prefix, dm, iteration, layer, count):
    ranks = list(range(16))
    filename = f'cache/{prefix}-l{layer:02d}-it{iteration}.pkl'
    with open(filename, 'rb') as f:
        flow_mats = pickle.load(f)
    res = []
    gains = []
    for i in range(count):
        flow_mat = torch.from_numpy(np.array(flow_mats[i]))
        world_size = 64 if prefix == 'tianhe' else 16
        lec = flow_mat.reshape(world_size, world_size, 1)
        gec = flow_mat.transpose(1, 0).reshape(world_size, world_size, 1)
        ne = 1
        exp, gain = policy_fn(lec, gec, ne, world_size, dm, True)
        res.append(exp.reshape(-1).tolist())
        gains.append(gain)
    return res, gains


# In[155]:


iters = [5500]
dms = [1024, 4096]

def get_estm(prefix, dm, iteration, layer):
    with open(f'sims/{dm}/{prefix}-l{layer}-it{iteration}.pkl', 'r') as f:
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
            
            filename = 'logs/{}/times-chaosflow/dm{}-l{}-it{}.pkl'.format(prefix, dm, layer, i)
            with open(filename, 'rb') as f:
                t = pickle.load(f)
            tov = np.array(t).sum(axis=1)
            
            repl, gain = get_repl(prefix, dm, i, layer, len(tov))
            results.append((tbl[2:], tov[2:], repl[2:], gain[2:]))
    return results # [np.concatenate(s) for s in zip(*results)]


# In[158]:


d = get_time('moe-bert',  1024)


# In[207]:


tbl, tov, repl, gain = d[12]

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

fig.set_size_inches(8, 3)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax2.set_ylabel('Experts')
z = np.array(repl).transpose(1, 0)
x = np.arange(z.shape[1] + 1)
y = np.arange(z.shape[0] + 1)
z = z
ax2.pcolormesh(x, y, z, vmin=0, vmax=1, cmap=plt.get_cmap('binary')) #, edgecolors='grey')
ax2.set_ylim(0, 16)
for i in range(1, z.shape[0] + 1):
    ax2.plot([0, z.shape[1]], [i, i], linewidth=1, color='grey')

ax.set_ylabel('Latency/ms')
ax.plot(tbl, marker='o', markevery=10, label='No opt.', markersize=8)
ax.plot(tov, marker='*', markevery=10, label='Shadowing', markersize=10)
ax.bar([0], [0], color='black', label='Shadow expert')
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.1), ncol=3)
# ax.plot(tbl - np.array(gain) * 1000, marker='x', linestyle='--', markevery=10)
ax.set_ylim(40, 120)
ax.set_xlim(0, 62)
ax.set_xticks([0, 62])
ax.set_xticklabels(['5000', '5062'])
ax2.set_xlabel('Iterations in layer 12, MoE-BERT-Deep')
ax2.set_yticks([0, 15])
plt.savefig('results/fig11.pdf', bbox_inches='tight')

