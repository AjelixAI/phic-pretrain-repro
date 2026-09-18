"""Synthetic biography facts (Allen-Zhu bioS-style) for the capacity probe."""
import torch, numpy as np

class BioSCorpus:
    def __init__(self, n_persons=1000, attrs=('birth', 'univ', 'work', 'month'),
                 values_per_attr=32, exposures=50, seed=0):
        rng = np.random.default_rng(seed)
        self.attrs = list(attrs)
        self.n_persons, self.vpa, self.exposures = n_persons, values_per_attr, exposures
        # vocab: 0 pad, 1..n_persons names, then REL_i, then value tokens per attr, then Q/SEP
        self.n_persons = n_persons
        self.rel_ids = {a: 1 + n_persons + i for i, a in enumerate(self.attrs)}
        base = 1 + n_persons + len(self.attrs)
        self.val_ids = {a: base + i * values_per_attr for i, a in enumerate(self.attrs)}
        self.vocab = base + len(self.attrs) * values_per_attr + 3
        self.Q, self.SEP, self.PAD = self.vocab - 3, self.vocab - 2, 0
        # facts: person -> attr -> value
        self.facts = {p: {a: int(rng.integers(values_per_attr)) for a in self.attrs}
                      for p in range(n_persons)}
        self.fact_list = [(p, a, self.facts[p][a]) for p in range(n_persons) for a in self.attrs]

    def statement(self, p, a):
        return [1 + p, self.rel_ids[a], self.val_ids[a] + self.facts[p][a], self.SEP]

    def query(self, p, a):
        return [self.Q, 1 + p, self.rel_ids[a]]

    def train_stream(self):
        """One epoch: each person's statements, repeated `exposures` times,
        shuffled at the line level."""
        lines = []
        for p in range(self.n_persons):
            for a in self.attrs:
                lines.append(self.statement(p, a))
        lines = np.array(lines)
        out = []
        for _ in range(self.exposures):
            perm = np.random.permutation(len(lines))
            out.extend(lines[perm].tolist())
        return out  # list of token lists

    def eval_queries(self):
        """All facts as (query tokens, correct value token)."""
        return [(self.query(p, a), self.val_ids[a] + v) for p, a, v in self.fact_list]
