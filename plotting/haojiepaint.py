#!/usr/bin/env python
# coding: utf-8

# In[3]:


from types import CodeType
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm


import matplotlib
from numpy.lib.function_base import cov
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dirbase = 'figure/eval/'
ourSys = 'APE'
figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 3),
}
plt.rcParams.update(figsz)
hb = '\\\\//\\\\//'

color_def = [
    '#f4b183',
    '#ffd966',
    '#c5e0b4',
    '#bdd7ee',
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#cccccc",
    "#fccde5",
    "#b3de69",
    "#ffd92f",
    '#fc8d59',
    '#74a9cf',
    '#66c2a4',
    '#f4a143',
    '#ffc936',
    '#78c679',
]

hatch_def = [
    '//',
    '\\\\',
    'xx',
    '++',
    '--',
    '||',
    '..',
    'oo',
    '',
]

marker_def = [
    'o',
    'D',
    'H',
    'v',
    '*',
    '+',
    '^',
    'x',
]


# In[2]:


def test_color(color_vec=color_def, hatch_vec=hatch_def, marker_def=marker_def, path=''):
    figsz = {'figure.figsize': (8, 3)}
    plt.rcParams.update(figsz)
    N = len(color_vec)
    dat = [1 for i in range(N)]
    fig, ax = plt.subplots()
    W = 0.35
    ind = np.arange(N) - 2 * W
    ax.bar(ind, dat, color=color_vec, hatch=hatch_vec)
    bars = ax.patches
    hatches = []
    for i in range(N):
        hatches.append(hatch_vec[i % len(hatch_vec)])
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], hatch=hatches[i], label=str(i)) for i in range(N)]
    plt.legend(legend_handles, range(
        N), bbox_to_anchor=(1, 1), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    xvals = np.arange(N)
    ax.set_xticks(ind)
    ax.set_xticklabels(str(x) for x in xvals)
    plt.show()
    if path != '':
        fig.savefig(path, bbox_inches='tight')


# test_color(color_def, hatch_def)


def parse_loss(filename):
    with open(filename) as f:
        lines = f.readlines()
        loss_lines = filter(lambda x: "lm loss:" in x, lines)
        losses = map(lambda x: float(x.split("lm loss:")[1].split()[0]), loss_lines)
        return list(losses)


def getcolors(n, cscheme='gist_ncar'):
    cmap = cm.get_cmap(cscheme)
    return [cmap(i / n) for i in range(n)]
