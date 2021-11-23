#!/usr/bin/env python
# coding: utf-8

from haojiepaint import *

# In[2]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


def parse_log(filename):
    results = dict()
    with open(filename, 'r') as f:
        for l in f:
            if l.startswith('Running'):
                key = l.split(' trace ')[1].split('/')[-1].strip()
                results[key + ' Comp'] = []
                results[key + ' Comm'] = []
            elif l.startswith('Scatter'):
                results[key + ' Comm'].append((float(l.split(' ')[2]), float(l.split(' ')[5])))
            elif l.startswith('Computation'):
                results[key + ' Comp'].append((float(l.split(' ')[2]), float(l.split(' ')[5])))
    return results

def parse_overall(filename):
    results = dict()
    key = 0
    with open(filename, 'r') as f:
        for l in f:
            if l.startswith('D_MODEL'):
                key += 1
                results[key] = []
            elif l.startswith('Prediction'):
                lsp = l.split(' ')
                results[key].append((float(lsp[2]) + float(lsp[5]), float(lsp[8]) + float(lsp[11])))
    return results


# In[38]:


d = parse_log('logs/estm.log')
fig, axs = plt.subplots(ncols=2, sharey=True)
fig.set_size_inches(6, 3)
xmax = 0.
colors = getcolors(len(d))
for i, r in enumerate(d):
    if r.endswith('Comm'):
        ax = axs[1]
    else:
        ax = axs[0]
    estms = [x[0] for x in d[r]]
    reals = [x[1] for x in d[r]]
    if len(estms) > 0:
        xmax = max(xmax, *estms)
        xmax = max(xmax, *reals)
    ax.scatter(np.mean(estms), np.mean(reals), label=r, 
               color=None, marker=marker_def[i % len(marker_def)], s=60)
    # ax.text(estms[0], reals[0], r)
xmax=20
for ax in axs:
    ax.plot([0, xmax], [0, xmax], dashes=(2, 2), color='gray', zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax)
axs[0].set_xlabel('Comp. Estimation / ms')
axs[1].set_xlabel('Comm. Estimation / ms')
    
# axs[0].set_title('Communication')
# axs[1].set_title('Computation')
axs[0].set_ylabel('Real / ms')
# axs[0].legend()
plt.savefig('results/fig9a.pdf', bbox_inches='tight')


# In[34]:


d = parse_overall('logs/pred.log')
fig, ax = plt.subplots()
fig.set_size_inches(3, 3)
xmax = 0.
colors = getcolors(len(d), cscheme='prism')
for i, r in enumerate(d):
    estms = [x[0] for x in d[r]]
    reals = [x[1] for x in d[r]]
    if len(estms) > 0:
        xmax = max(xmax, *estms)
        xmax = max(xmax, *reals)
    ax.scatter(np.mean(estms), np.mean(reals), label=r,
               marker=marker_def[i % len(marker_def)], s=60)
    # ax.text(estms[0], reals[0], r)

xmax = 350
ax.plot([0, xmax], [0, xmax], dashes=(2, 2), color='gray', zorder=0)

ax.set_xlim(0, xmax)
ax.set_ylim(0, xmax)
ax.set_xlabel('Estimation / ms')
ax.set_ylabel('Real / ms')
# axs[0].legend()
plt.savefig('results/fig9b.pdf', bbox_inches='tight')


# You can compute R2 score using the following code
# from sklearn.metrics import r2_score
# 
# xs, ys = [], []
# for r in d:
#     estms = [x[0] for x in d[r]]
#     reals = [x[1] for x in d[r]]
#     xs += estms
#     ys += reals
# r2_score(xs, ys)

