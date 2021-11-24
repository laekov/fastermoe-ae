#!/usr/bin/env python
# coding: utf-8

from haojiepaint import *

import os
import pickle
from utils import processlabel

# In[481]:


def get_time(prefix, name, dm, layers=-1):
    iters = [500, 5500, 80500]
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
    for layer in la:
        layer_time = 0
        last_iter = 0
        for i in iters:
            filename = 'logs/{}/times-{}/dm{}-l{}-it{}.pkl'.format(prefix, name, dm, layer, i)
            # print(filename)
            if not os.path.exists(filename):
                if last_iter == 0:
                    return -1
                print('gg at {}'.format(i))
                return layer_time / last_iter * layers
            with open(filename, 'rb') as f:
                t = pickle.load(f)
            t = t[2:]
            t = np.array(t).sum(axis=1)
            t = t.mean(axis=0)
            # print(t, i, last_iter, (i - last_iter))
            layer_time += t * (i - last_iter)
            last_iter = i
        tot_time += layer_time
    return tot_time / last_iter



def get_deepspeed(prefix, stage, dm):
    if prefix == 'tianhe':
        return get_deepspeed_tianhe(stage, dm)
    bsz = 2048 if prefix.find('gpt') != -1 else 4096
    layers = 12 if prefix.find('gpt') != -1 else 24
    filename = 'logs/{}/times-ds-{}/bsz{}-dm{}-l0-it500.pkl'.format(prefix, stage, bsz, dm)
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    d = d[0][2:]
    t = np.array(d).sum(axis=1).mean(axis=0)
    return t * layers
            

prefixs = ['moe-gpt', 'moe-bert']
bare_tests = ['fastmoe', 'chaosflow']

sers = dict(fastmoe=[], chaosflow=[])
for t in range(1, 4):
    sers['ds-{}'.format(t)] = []


for prefix in prefixs:
    dms = [1024, 4096]
    for dm in dms:
        for t in range(1, 4):
            dsi = get_deepspeed(prefix, t, dm)
            sers['ds-{}'.format(t)].append(dsi)
        for tn in bare_tests:
            sers[tn].append( get_time(prefix, tn, dm))


fig, ax = plt.subplots()
fig.set_size_inches(8, 3)

xbase = np.arange(len(dms) * len(prefixs))

baseline = np.array(sers['ds-3'])
ser_keys = ['ds-{}'.format(i) for i in range(1, 4)] + bare_tests
wid = .8 / len(ser_keys)

for i, ser in enumerate(ser_keys):
    su = baseline / np.array(sers[ser])
    # print(ser, max(su[:4]), max(su[4:]))
    ax.bar(xbase + i * wid, su, label=processlabel(ser), width=wid,
           color=color_def[i], hatch=hatch_def[i], edgecolor='black')
ax.set_xticks(xbase + len(ser_keys) * wid * .5)
ax.legend() # loc='lower left', bbox_to_anchor=(0, 1), ncol=2)
ax.set_ylabel('Speedup')
#ax.set_xticklabels(sers.keys())

ax.set_ylim(0, 10)
ax.plot([4 - wid, 4 - wid], [0, 6], color='black', linestyle='--', linewidth=1)

ax.set_xticklabels(['GPT-S', 'GPT-L', 'BERT-Deep', 'BERT-Deep-L'])

plt.savefig('results/fig10.pdf', bbox_inches='tight')

