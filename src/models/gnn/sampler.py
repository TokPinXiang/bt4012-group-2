"""Neighbourhood sampler: picks the part of the graph the GNN sees for each payment.

The full graph is too large to process at once, so each scored payment gets a small subgraph: up to 100 payments
touching its sender or receiver, then up to 100 more around each of those accounts (two hops; sizes set by
`fanout`). The payments being scored are always included. In time-respecting mode (`max_time`), only payments made
up to the scored payment's minute can be picked, so the model never sees later payments.
"""
import numpy as np
import torch


class EdgeSampler:
    def __init__(self, edge_index, n_nodes, allowed_edges, seed=0, ts=None):
        """allowed_edges: bool mask of the edges that exist in this graph view (train / train+val / all)."""
        ei = edge_index.numpy()
        eid = np.where(allowed_edges)[0]
        nodes = np.concatenate([ei[0, eid], ei[1, eid]])
        edges = np.concatenate([eid, eid])
        self.ts = None
        if ts is not None:  # incidences sorted by (node, time), so a time cut is a prefix per node
            self.ts = np.asarray(ts, dtype=np.int64)
            self.t0 = self.ts.min()
            t_rel = self.ts - self.t0
            order = np.lexsort((t_rel[edges], nodes))
            self.comp = nodes[order].astype(np.int64) * (1 << 32) + t_rel[edges][order]
        else:
            order = np.argsort(nodes, kind="stable")
        self.inc = edges[order]
        self.ptr = np.zeros(n_nodes + 1, np.int64)
        np.add.at(self.ptr, nodes + 1, 1)
        self.ptr = np.cumsum(self.ptr)
        self.ei = ei
        self.rng = np.random.default_rng(seed)

    def sample(self, target_eids, fanout, max_time=None):
        tgt = np.asarray(target_eids)
        frontier = np.unique(self.ei[:, tgt].ravel())
        seen = frontier
        edge_set = []
        for f in fanout:
            if max_time is not None and self.ts is not None:
                q = frontier.astype(np.int64) * (1 << 32) + (int(max_time) - self.t0)
                deg = np.searchsorted(self.comp, q, side="right") - self.ptr[frontier]
            else:
                deg = self.ptr[frontier + 1] - self.ptr[frontier]
            ar = np.arange(f)[None, :]
            rand = np.floor(self.rng.random((len(frontier), f)) * deg[:, None]).astype(np.int64)
            pos = np.where(deg[:, None] <= f, ar, rand)
            valid = ar < deg[:, None]
            e = np.unique(self.inc[(self.ptr[frontier][:, None] + pos)[valid]])
            edge_set.append(e)
            nb = np.unique(self.ei[:, e].ravel())
            frontier = np.setdiff1d(nb, seen, assume_unique=True)
            seen = np.union1d(seen, nb)
            if len(frontier) == 0:
                break
        ctx = np.setdiff1d(np.concatenate(edge_set) if edge_set else np.array([], np.int64), tgt)
        eids = np.concatenate([tgt, ctx])  # targets first
        nid, inv = np.unique(self.ei[:, eids].ravel(), return_inverse=True)
        return (torch.from_numpy(eids), torch.from_numpy(nid), torch.from_numpy(inv.reshape(2, -1).astype(np.int64)),
                len(tgt))
