"""The GNN: gives each payment a laundering score from its sampled neighbourhood (accounts = nodes, payments = edges).

1. Start: each payment begins as its features; each account starts blank, except that the two accounts of the
   payment being scored are marked (ego IDs).
2. Two rounds of message passing: each account collects information from the payments around it, both money going
   out and money coming in (reverse message passing). After each round, each payment is updated from its two
   accounts (edge updates).
3. Optional pooling (pair_agg): all payments between the same two accounts are combined into one message (mean,
   max and count), so a pair that trades many times does not swamp the rest.
4. Score: the sender, the receiver and the payment itself are joined and a small network outputs the score.

Accounts combine messages with GIN (a sum) or PNA (mean, min, max and spread, adjusted for how busy the account is).
Each step can be switched off for the ablation. Based on IBM Multi-GNN (Egressy et al., AAAI 2024, Apache-2.0) and
MEGA-GNN (arXiv 2412.00241).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GINEConv, PNAConv


def make_conv(kind, h, deg=None):
    if kind == "gin":
        return GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h)
    if kind == "pna":
        return PNAConv(h, h, aggregators=["mean", "min", "max", "std"],
                       scalers=["identity", "amplification", "attenuation"], deg=deg, edge_dim=h,
                       towers=5, pre_layers=1, post_layers=1, divide_input=False)
    raise ValueError(kind)


class EdgeGNN(nn.Module):
    def __init__(self, edge_dim, kind="gin", hidden=64, layers=2, edge_updates=True, reverse_mp=True,
                 ego=True, dropout=0.1, final_dropout=0.3, deg=None, pair_agg=False):
        super().__init__()
        if kind == "pna":
            hidden = (hidden // 5) * 5  # PNA towers need hidden divisible by 5
        self.layers, self.eu, self.rmp, self.ego, self.pair_agg = layers, edge_updates, reverse_mp, ego, pair_agg
        self.dropout = dropout
        self.node_emb = nn.Linear(2 if ego else 1, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)
        self.fwd = nn.ModuleList([make_conv(kind, hidden, deg) for _ in range(layers)])
        self.rev = nn.ModuleList([make_conv(kind, hidden, deg) for _ in range(layers)]) if reverse_mp else None
        self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers)])
        self.emlps = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                                    for _ in range(layers)]) if edge_updates else None
        self.pair_mlp = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden)) if pair_agg else None
        self.head = nn.Sequential(nn.Linear(3 * hidden, 64), nn.ReLU(), nn.Dropout(final_dropout),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(final_dropout), nn.Linear(32, 2))

    def forward(self, edge_index, edge_attr, n_nodes, target_nodes=None, n_out=None):
        x0 = torch.ones(n_nodes, 1, device=edge_attr.device)
        if self.ego:
            ego = torch.zeros(n_nodes, 1, device=edge_attr.device)
            if target_nodes is not None:
                ego[target_nodes] = 1.0
            x0 = torch.cat([x0, ego], 1)
        x = self.node_emb(x0)
        e = self.edge_emb(edge_attr)
        src, dst = edge_index
        if self.pair_agg:
            uniq, inv = torch.unique(src * n_nodes + dst, return_inverse=True)
            mp_index = torch.stack([uniq // n_nodes, uniq % n_nodes])
            cnt = torch.zeros(len(uniq), device=e.device).index_add_(0, inv, torch.ones_like(inv, dtype=e.dtype))
        else:
            mp_index = edge_index
        rev_index = mp_index.flip(0)
        for i in range(self.layers):
            if self.pair_agg:
                mean = torch.zeros(len(uniq), e.shape[1], device=e.device).index_add_(0, inv, e) / cnt[:, None]
                mx = torch.full((len(uniq), e.shape[1]), -1e9, device=e.device).scatter_reduce(
                    0, inv[:, None].expand_as(e), e, reduce="amax")
                me = self.pair_mlp(torch.cat([mean, mx, torch.log1p(cnt)[:, None]], -1))
            else:
                me = e
            h = self.fwd[i](x, mp_index, me)
            if self.rmp:
                h = (h + self.rev[i](x, rev_index, me)) / 2
            x = (x + F.dropout(F.relu(self.bns[i](h)), self.dropout, self.training)) / 2
            if self.eu:
                e = e + self.emlps[i](torch.cat([x[src], x[dst], e], -1)) / 2
        if n_out is not None:  # only the first n_out edges are targets
            src, dst, e = src[:n_out], dst[:n_out], e[:n_out]
        return self.head(torch.cat([x[src], x[dst], e], -1))
