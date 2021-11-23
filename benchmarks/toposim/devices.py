class Link(object):
    def __init__(self, dst, bw, duplex=None):
        self.bw = bw
        self.dst = dst
        self.duplex = duplex
        if duplex is not None:
            duplex.duplex = self
        self.traffic = 0

class Device(object):
    def __init__(self, name):
        self.name = name
        self.parent_link = None
        self.child_links = []
        self.peer_links = []
        self.peers = dict()
        self.depth = -1 


class Cluster(object):
    def __init__(self):
        self.devices = []
        self.links = []

    def create_tree(self, root):
        root.depth = 0
        stack = [root]
        while len(stack) > 0:
            u = stack.pop()
            for e in u.child_links:
                self.links.append(e)
                self.links.append(e.duplex)
                v = e.dst
                v.depth = u.depth + 1
                stack.append(v)

    def create_pcie_gpu_cluster(self, nnodes, ncpus_per_node, ngpus_per_cpu,
            bw_net=50e9 / 8, bw_pcie=88e9 / 8, bw_qpi=180e9 / 8):
        self.ibs = []
        self.cpus = []
        self.gpus = []
        self.switch = Device('root_switch')
        for node_rank in range(nnodes):
            ib = Device('node{}:ib'.format(node_rank))
            ib.parent_link = Link(self.switch, bw_net)
            self.switch.child_links.append(Link(ib, bw_net, ib.parent_link))
            self.ibs.append(ib)
            for cpu_rank in range(ncpus_per_node):
                cpu = Device('Node{}:cpu{}'.format(node_rank, cpu_rank))
                if cpu_rank == 0:
                    bw = -1
                else:
                    bw = bw_qpi
                cpu.parent_link = Link(ib, bw)
                ib.child_links.append(Link(cpu, bw, cpu.parent_link))
                self.cpus.append(cpu)
                for gpu_rank in range(ngpus_per_cpu):
                    gpu = Device('Node{}:gpu{}-{}'
                            .format(node_rank, cpu_rank, gpu_rank))
                    gpu.parent_link = Link(cpu, bw_pcie)
                    cpu.child_links.append(Link(gpu, bw_pcie, gpu.parent_link))
                    self.gpus.append(gpu)
        self.create_tree(self.switch)


    def add_peer(self, u, v, bw):
        u = self.gpus[u]
        v = self.gpus[v]

        link = Link(v, bw)
        u.peers[v] = len(u.peer_links)
        u.peer_links.append(link)
        self.links.append(link)

        link = Link(u, bw, link)
        v.peers[u] = len(v.peer_links)
        v.peer_links.append(link)
        self.links.append(link)


    def create_nvlink_cluster(self, nnodes, bw_net=62e9 / 8,
            bw_pix=55e9 / 8, bw_netnode=62e9 / 8,
            bw_nv1=171e9 / 8, bw_nv2=317e9 / 8, bw_node=88e9 / 8):
        self.ibs = []
        self.gpus = []
        self.switch = Device('root_switch')
        for node_rank in range(nnodes):
            ib = Device('node{}:ib'.format(node_rank))
            ib.parent_link = Link(self.switch, bw_net)
            self.switch.child_links.append(Link(ib, bw_net, ib.parent_link))
            self.ibs.append(ib)
            for gpu_rank in range(4):
                gpu = Device('Node{}:gpu{}'.format(node_rank, gpu_rank))
                if gpu_rank < 2:
                    bw = bw_pix
                else:
                    bw = bw_netnode
                gpu.parent_link = Link(ib, bw)
                ib.child_links.append(Link(gpu, bw, gpu.parent_link))
                self.gpus.append(gpu)
        self.create_tree(self.switch)
        for node_rank in range(nnodes):
            idx = node_rank * 4
            self.add_peer(idx + 0, idx + 1, bw_nv2)
            self.add_peer(idx + 2, idx + 3, bw_nv2)
            self.add_peer(idx + 0, idx + 3, bw_nv1)
            self.add_peer(idx + 1, idx + 2, bw_nv1)
            self.add_peer(idx + 0, idx + 2, bw_node)
            self.add_peer(idx + 1, idx + 3, bw_node)


    def reset_traffic(self):
        for l in self.links:
            l.traffic = 0


    def add_traffic(self, u, v, traffic):
        r"""
        A flow from u to v with traffic bytes
        Using LCA algorithm
        """
        if isinstance(u, int):
            u = self.gpus[u]
            v = self.gpus[v]
        if v in u.peers:
            link = u.peer_links[u.peers[v]]
            link.traffic += traffic
        while u != v:
            if u.depth > v.depth:
                u.parent_link.traffic += traffic
                u = u.parent_link.dst
            else:
                v.parent_link.duplex.traffic += traffic
                v = v.parent_link.dst

    def get_latency(self):
        latency = 0
        for l in self.links:
            if l.bw != -1:
                latency = max(latency, l.traffic / l.bw)
        return latency
