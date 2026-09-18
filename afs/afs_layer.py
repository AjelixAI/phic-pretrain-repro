"""Associative Factor Store (AFS) — the retrieval-FFN layer.
v1 rows: dense mini-FFNs (the clean capacity baseline; bits/param measurement).
v2 rows (next): factorized tied-blocks rows sharing the layer's frozen basis
   (optics-renderable; measures the structured tax vs v1 directly).
Retrieval: scores = x @ K^T; soft (train, all rows) or hard top-k + straight-
through (serve — reads only k rows: the bandwidth solve)."""
import torch, torch.nn as nn, torch.nn.functional as F

class AFSFFN(nn.Module):
    def __init__(self, d, f_row, E, hard_k=0, dropout=0.0):
        """d: model width; f_row: per-row FFN hidden width; E: rows;
        hard_k: >0 -> hard top-k with straight-through soft softmax."""
        super().__init__()
        self.d, self.E, self.hard_k = d, E, hard_k
        self.keys = nn.Parameter(torch.randn(E, d) / d ** 0.5)
        self.rows = nn.ModuleList()
        for _ in range(E):
            self.rows.append(nn.Sequential(
                nn.Linear(d, f_row, bias=False), nn.SiLU(),
                nn.Linear(f_row, f_row, bias=False), nn.SiLU(),
                nn.Linear(f_row, d, bias=False)))
        self.drop = nn.Dropout(dropout)

    def scores(self, x):                       # [B,S,E]
        return (x @ self.keys.T) / self.keys.shape[1] ** 0.5

    def forward(self, x):
        s = self.scores(x)
        if self.hard_k and self.hard_k < self.E:
            topv, topi = s.topk(self.hard_k, dim=-1)
            w = topv.softmax(-1)
            w_st = w + (s.softmax(-1).gather(-1, topi) - w).detach()  # straight-through
            out = 0.0
            for j in range(self.hard_k):       # reads only k rows (the bandwidth solve)
                idx = topi[..., j]             # [B,S]
                wj = w_st[..., j]
                # gather per-token selected rows: group by row id for efficiency
                out = out + self._apply_gathered(x, idx, wj)
            return self.drop(out)
        w = s.softmax(-1)                      # soft: all rows (training path)
        out = 0.0
        for e, row in enumerate(self.rows):
            out = out + w[..., e:e + 1] * row(x)
        return self.drop(out)

    def _apply_gathered(self, x, idx, wj):
        # per-token row selection without materializing all rows:
        # compute per unique row id in this batch (small k keeps this cheap)
        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        wflat = wj.reshape(-1)
        xflat = x.reshape(-1, x.shape[-1])
        for e in flat.unique():
            m = flat == e
            y = self.rows[e](xflat[m])
            out.view(-1, out.shape[-1])[m] = y * wflat[m][:, None]
        return out

class ProductKeyAFS(AFSFFN):
    """Product-key retrieval (T4): row (i,j) keyed by k1_i + k2_j.
    Table size E1*E2 rows; scoring costs (E1+E2) dot products (T4/T8).
    Rows are stored as a flat ModuleList of E1*E2 operators."""
    def __init__(self, d, f_row, E1, E2, hard_k=0, dropout=0.0):
        super().__init__(d, f_row, E1 * E2, hard_k=0, dropout=dropout)
        self.E1, self.E2 = E1, E2
        self.keys = None
        self.k1 = nn.Parameter(torch.randn(E1, d) / d ** 0.5)
        self.k2 = nn.Parameter(torch.randn(E2, d) / d ** 0.5)
        self.hard_k = hard_k

    def scores(self, x):                       # [B,S,E1,E2] via two matvecs
        s1 = x @ self.k1.T                      # [B,S,E1]
        s2 = x @ self.k2.T                      # [B,S,E2]
        return s1[..., :, None] + s2[..., None, :]

    def forward(self, x):
        s = self.scores(x).reshape(*x.shape[:-1], self.E1 * self.E2)
        if self.hard_k and self.hard_k < self.E1 * self.E2:
            topv, topi = s.topk(self.hard_k, dim=-1)
            w = topv.softmax(-1)
            w = w + (s.softmax(-1).gather(-1, topi) - w).detach()
            out = torch.zeros_like(x)
            flat_i = topi.reshape(-1); w_f = w.reshape(-1)
            x_f = x.reshape(-1, x.shape[-1]); o_f = out.view(-1, out.shape[-1])
            for e in flat_i.unique():
                m = flat_i == e
                o_f[m] = self.rows[e](x_f[m]) * w_f[m][:, None]
            return self.drop(out)
        w = s.softmax(-1)
        out = 0.0
        for e, row in enumerate(self.rows):
            out = out + w[..., e:e + 1] * row(x)
        return self.drop(out)
