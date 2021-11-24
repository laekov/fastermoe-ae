import os
import sys
import numpy as np


def gen_groups(a, b, c):
    ranks = np.arange(a * b * c).reshape(a, b, c)
    x = ranks.transpose(1, 2, 0).reshape(-1, a).tolist()
    y = ranks.transpose(0, 2, 1).reshape(-1, b).tolist()
    z = ranks.reshape(-1, c).tolist()
    return x, y, z


def enumerate_groups(world_size):
    facs = set()
    for i in range(1, world_size + 1):
        if world_size % i == 0:
            facs.add(i)
            facs.add(world_size // i)
            if i * i >= world_size:
                break
    grps = set()
    for a in facs:
        for b in facs:
            c = world_size // (a * b)
            if a * b * c == world_size:
                x, y, z = gen_groups(a, b, c)
                grps.add(str(dict(dp=x, mp=y, moe=z)))
                grps.add(str(dict(dp=x, mp=z, moe=y)))
                grps.add(str(dict(dp=y, mp=x, moe=z)))
                grps.add(str(dict(dp=y, mp=z, moe=x)))
                grps.add(str(dict(dp=z, mp=x, moe=y)))
                grps.add(str(dict(dp=z, mp=y, moe=x)))
    for g in grps:
        yield eval(g)


def main(world_size):
    if os.environ.get('CHAOS_MODE', '') == 'enum':
        for g in enumerate_groups(world_size):
            print(g)
        return
    mp_size = 2
    moe_size = 8
    dp_size = world_size // (mp_size * moe_size)
    ranks = np.arange(world_size).reshape(mp_size, moe_size, dp_size)
    d, m, e = gen_groups(mp_size, moe_size, dp_size)
    groups = dict(dp=d, mp=m, moe=e)
    print(groups)


if __name__ == '__main__':
    world_size = 16
    if len(sys.argv) > 1:
        world_size = int(sys.argv[1])
    d = main(world_size)
