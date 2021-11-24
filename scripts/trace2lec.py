import os
import sys
import numpy as np
import pickle


def get_lec(gate, world_size):
    lec = [0] * world_size
    for v in np.array(gate).reshape(-1):
        if v != -1:
            lec[v] += 1
    return lec


def parse_trace(path, iteration, layer, world_size):
    if iteration == 500:
        baseit = 1
    else:
        baseit = iteration - 500
    lec = []
    for rank in range(world_size):
        trace_path = '{}/{}/{}.pkl'.format(path, rank, iteration)
        with open(trace_path, 'rb') as f:
            data = pickle.load(f)
        for l, k in data:
            if l == layer:
                gate_idx = data[l, k]
                i = k - baseit
                if len(lec) <= i:
                    lec.append([])
                lec[i].append(get_lec(gate_idx, world_size))
    return lec


if __name__ == '__main__':
    trace_path = sys.argv[1]
    trace_name = trace_path.split('/')[-1]

    layer = int(sys.argv[2])
    iteration = int(sys.argv[3])

    t = parse_trace(trace_path, iteration, layer, 16)

    lec_path = 'cache/{}-l{:02d}-it{}.pkl'.format(
            trace_name.split('/')[-1], layer, iteration)
    with open(lec_path, 'wb') as f:
        pickle.dump(t, f)
    print('dumped {}'.format(lec_path))
