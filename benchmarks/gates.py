import torch
import pickle
from fmoe.gates.base_gate import BaseGate


class AllGoGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, target=0):
        super().__init__(num_expert, world_size)
        self.target = target 
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        assert not return_all_scores
        topk_idx = torch.ones(inp.shape[0],
                dtype=torch.long, device=inp.device) * self.target
        topk_score = torch.ones(inp.shape[0], self.top_k,
                device=inp.device) / self.top_k
        return topk_idx, topk_score


class FlatGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k):
        super().__init__(num_expert, world_size)
        self.world_size = world_size
        self.top_k = top_k

    def forward(self, inp, _=False):
        assert inp.shape[0] * self.top_k % self.world_size == 0
        idx = torch.arange(self.world_size, device=inp.device, dtype=torch.long)
        idx = idx.repeat(inp.shape[0] * self.top_k // self.world_size)
        score = torch.ones(inp.shape[0], self.top_k,
                device=inp.device) / self.top_k
        return idx, score


class RandomGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, option):
        super().__init__(num_expert, world_size)
        self.world_size = world_size
        self.top_k = top_k
        self.option = option

    def forward(self, inp, _=False):
        assert inp.shape[0] * self.top_k % self.world_size == 0
        if self.option == -2:
            lb = -1
            ub = self.tot_expert
        elif self.option == -3:
            lb = 0
            ub = self.tot_expert
        else:
            assert False
        idx = torch.randint(lb, ub, (inp.shape[0], self.top_k)).cuda()
        score = torch.ones(inp.shape[0], self.top_k,
                device=inp.device) / self.top_k
        return idx, score


class TraceRepGate(BaseGate):
    def __init__(self, path, layer, ranks, world_size, iteration):
        if isinstance(ranks, int):
            ranks = [ranks]
        self.world_size = world_size
        data_on_ranks = []
        for rank in ranks:
            trace_path = '{}/{}/{}.pkl'.format(path, rank, iteration)
            with open(trace_path, 'rb') as f:
                data = pickle.load(f)
            data_of_rank = []
            for l, i in data:
                if l == layer:
                    gate_idx = torch.Tensor(data[l, i]).long().cuda()
                    mask = (gate_idx == -1)
                    gate_idx = gate_idx % world_size
                    gate_idx[mask] = -1
                    data_of_rank.append(gate_idx)
            data_on_ranks.append(data_of_rank)

        self.data = []
        for d in zip(*data_on_ranks):
            self.data.append(torch.cat(d))

        self.i = 0
        self.top_k = self.data[0].shape[1]
        self.batch_size = self.data[0].shape[0]
        super().__init__(1, world_size)

    def forward(self, inp, _=False):
        idx = self.data[self.i]
        score = torch.ones(inp.shape[0], self.top_k,
                device=inp.device) / self.top_k
        return idx, score

    def next(self):
        self.i = (self.i + 1) % len(self.data)

    def current(self):
        return self.data[self.i]
