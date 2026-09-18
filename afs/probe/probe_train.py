"""Capacity probe: train a small LM with AFS or dense FFN on bioS facts,
measure extractable recall -> bits/trained-param.
Arms: --ffn afs --E 32|128|512 | --ffn dense --f-row <compute-matched>"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math, sys, os, numpy as np
sys.path.insert(0, '/tmp/phic-pretrain-repro')
sys.path.insert(0, '/tmp/phic-pretrain-repro/afs')
sys.path.insert(0, '/tmp/phic-pretrain-repro/afs/probe')
from afs_layer import AFSFFN, ProductKeyAFS
from bios import BioSCorpus

class Block(nn.Module):
    def __init__(self, d, nh, ffn, args):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.nh = nh
        if args.ffn == 'pk':
            E2 = args.E2 or int(args.E ** 0.5)
            self.ffn = ProductKeyAFS(d, args.f_row, max(1, args.E // E2), E2, hard_k=args.hard_k)
        elif args.ffn == 'afs':
            self.ffn = AFSFFN(d, args.f_row, args.E, hard_k=args.hard_k)
        else:
            self.ffn = nn.Sequential(nn.Linear(d, args.dense_f, bias=False), nn.SiLU(),
                                     nn.Linear(args.dense_f, d, bias=False))
    def forward(self, x):
        B, S, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        a = F.scaled_dot_product_attention(
            q.view(B, S, self.nh, -1).transpose(1, 2),
            k.view(B, S, self.nh, -1).transpose(1, 2),
            v.view(B, S, self.nh, -1).transpose(1, 2), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, d))
        return x + self.ffn(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab, args.d)
        self.blocks = nn.ModuleList([Block(args.d, args.nh, None, args) for _ in range(args.L)])
        self.nf = nn.LayerNorm(args.d)
        self.head = nn.Linear(args.d, args.vocab, bias=False)
    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks: x = b(x)
        return self.head(self.nf(x))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ffn', default='afs', choices=['afs', 'dense', 'pk'])
    p.add_argument('--E', type=int, default=4096)
    p.add_argument('--E2', type=int, default=0)
    p.add_argument('--f-row', type=int, default=256)
    p.add_argument('--dense-f', type=int, default=1024)
    p.add_argument('--hard-k', type=int, default=0)
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--L', type=int, default=4)
    p.add_argument('--nh', type=int, default=4)
    p.add_argument('--n-persons', type=int, default=1000)
    p.add_argument('--exposures', type=int, default=50)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dev', default='cuda:0')
    p.add_argument('--tag', default='probe')
    args = p.parse_args()
    torch.manual_seed(0)
    corp = BioSCorpus(n_persons=args.n_persons, exposures=args.exposures)
    args.vocab = corp.vocab
    m = TinyLM(args).bfloat16().to(args.dev)
    npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ffn_par = sum(p.numel() for n_, p in m.named_parameters() if 'ffn' in n_ or 'keys' in n_)
    print(f"[{args.tag}] ffn={args.ffn} E={args.E} | trained {npar/1e6:.2f}M (ffn-store {ffn_par/1e6:.2f}M)", flush=True)
    lines = corp.train_stream()
    flat = [t for l in lines for t in l + [corp.SEP]]
    T = torch.tensor(flat, device=args.dev)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(m.parameters(), lr=args.lr, weight_decay=0.01)
    except ImportError:
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * (1 - s / args.steps * 0.9))
    def chunks():
        for i in range(0, T.numel() - args.bs * 256 - 1, args.bs * 256):
            c = T[i:i + args.bs * 256 + 1]
            yield c[:-1].view(args.bs, 256), c[1:].view(args.bs, 256)
    data = list(chunks())
    step, t0 = 0, __import__('time').time()
    rng = np.random.default_rng(0)
    while step < args.steps:
        for _ in range(len(data)):
            if step >= args.steps: break
            xb, yb = data[rng.integers(len(data))]   # random chunk per step: no sequence memorization
            logits = m(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 500 == 0:
                print(f"[{args.tag}] step {step} loss {loss.item():.4f}", flush=True)
            step += 1
    # recall eval: every fact
    m.eval()
    qs, ys = zip(*corp.eval_queries())
    L = max(len(q) for q in qs)
    X = torch.zeros(len(qs), L, dtype=torch.long, device=args.dev)
    for i, q in enumerate(qs): X[i, :len(q)] = torch.tensor(q)
    M = torch.tensor([len(q) - 1 for q in qs], device=args.dev)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(qs), 256):
            lg = m(X[i:i+256])
            pred = lg[torch.arange(len(X[i:i+256])), M[i:i+256]].argmax(-1)
            correct += (pred.cpu() == torch.tensor(ys[i:i+256])).sum().item()
    n_facts = len(qs)
    bits_per_fact = math.log2(corp.vpa)
    recall = correct / n_facts
    capacity_bits = recall * n_facts * bits_per_fact
    print(f"[{args.tag}] RESULT recall {recall:.4f} ({correct}/{n_facts}) | "
          f"capacity {capacity_bits/1e6:.3f} Mbit | bits/ffn-param {capacity_bits/ffn_par:.4f} | "
          f"trained {npar/1e6:.2f}M", flush=True)
    torch.save({'args': vars(args), 'model': m.state_dict(), 'recall': recall},
               f"/tmp/afs_probe_{args.tag}.pt")

if __name__ == '__main__':
    main()
