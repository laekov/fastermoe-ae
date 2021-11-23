def alltoall(cluster, ranks, flow_matrix, dsize=4):
    assert len(ranks) == len(flow_matrix)
    world_size = len(ranks)
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                cluster.add_traffic(ranks[i], ranks[j],
                        flow_matrix[i][j] * dsize)


def _ring(cluster, ranks, traffic):
    for i in range(len(ranks)):
        cluster.add_traffic(ranks[i - 1], ranks[i], traffic)


def allreduce(cluster, ranks, nele, dsize=4):
    # Assume ring allreduce
    n = len(ranks)
    traffic = dsize * 2 * (n - 1) / n * nele
    _ring(cluster, ranks, traffic)


def bcast(cluster, ranks, nele, dsize=4):
    n = len(ranks)
    traffic = dsize * nele
    _ring(cluster, ranks, traffic)


def redus(cluster, ranks, nele, dsize=4):
    # This is not a spelling error.
    # Rename reduce as redus to avoid keyword conflict.
    n = len(ranks)
    traffic = dsize * nele
    _ring(cluster, ranks, traffic)
