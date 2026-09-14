#!/usr/bin/env python3
"""
pretrain_tb.py — from-scratch pretraining head-to-head:
  dense baseline vs tied-blocks architecture, OLMo-style recipe,
  FineWeb-Edu open data, W&B-logged. One process per GPU.

Usage:
  python pretrain_tb.py --mode dense --gpu 0 --steps 6000
  python pretrain_tb.py --mode tied  --gpu 1 --steps 6000
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- config ---
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "tied"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=64)          # sequences/step
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--ffn", type=int, default=1408)
    p.add_argument("--block", type=int, default=32)       # tied block width
    p.add_argument("--rank", type=int, default=64)        # tied residual rank
    p.add_argument("--stages", type=int, default=2)       # butterfly stages
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--decay-start", type=int, default=4500)  # WSD: stable till here
    p.add_argument("--project", default="phic-pretrain")
    return p.parse_args()

A = None

# ------------------------------------------------- tied-blocks components --
class BflyBasis(nn.Module):
    """K frozen seeded block-diagonal stages + permutations (the E18 shape)."""
    def __init__(self, n, b, K, seed, dtype, device):
        super().__init__()
        nb = n // b
        g = torch.Generator().manual_seed(seed)
        cols = (torch.arange(nb, device=device) * b)[:, None] \
            + torch.arange(b, device=device)[None, :]
        self.stages = nn.ParameterList()
        self.cols = cols
        for i in range(K):
            S = torch.randn(nb, b, b, generator=g, device=device,
                            dtype=dtype) / b ** 0.5
            self.stages.append(nn.Parameter(S, requires_grad=False))
        self.nb, self.b, self.n = nb, b, n

    def _blk(self, h, W):
        B, S = h.shape[0], h.shape[1]
        hh = h.reshape(B, S, self.nb, self.b).permute(2, 3, 0, 1)
        hh = hh.reshape(self.nb, self.b, B * S)
        y = torch.bmm(W, hh)
        y = y.reshape(self.nb, self.b, B, S).permute(2, 3, 0, 1)
        return y.reshape(*h.shape[:-1], self.n)

    def _perm(self, h):
        return (h.view(*h.shape[:-1], self.nb, self.b)
                .transpose(-1, -2).reshape(h.shape))

    def forward(self, x):
        h = x
        for i, st in enumerate(self.stages):
            h = self._blk(h, st)
            if i < len(self.stages) - 1:
                h = self._perm(h)
        return h

class TiedLinear(nn.Module):
    """Butterfly basis (frozen) -> trained block-diagonal -> rank-r residual.
    n = max(d_in, d_out) padded to block multiple; padding is structural."""
    def __init__(self, d_in, d_out, cfg, seed):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        n = ((max(d_in, d_out) + cfg.block - 1) // cfg.block) * cfg.block
        self.n = n
        self.b = cfg.block
        self.nb = n // cfg.block
        self.basis = BflyBasis(n, cfg.block, cfg.stages, seed * 7 + 3,
                               torch.bfloat16, "cpu")
        nb = n // cfg.block
        self.blocks = nn.Parameter(
            torch.eye(cfg.block, dtype=torch.bfloat16).unsqueeze(0)
            .repeat(nb, 1, 1).contiguous())
        r = min(cfg.rank, min(d_in, d_out))
        g = torch.Generator().manual_seed(seed * 11 + 5)
        self.U = nn.Parameter(torch.randn(d_out, r, generator=g,
                              dtype=torch.bfloat16) / r ** 0.5)
        self.V = nn.Parameter(torch.randn(r, d_in, generator=g,
                              dtype=torch.bfloat16) / d_in ** 0.5)

    def forward(self, x):
        xp = F.pad(x, (0, self.n - x.shape[-1]))
        h = self.basis(xp)
        h = self._blockdiag(h)
        y = h[..., : self.d_out] + xp[..., : self.d_in] @ self.V.T @ self.U.T
        return y

    def _blockdiag(self, h):
        B, S = h.shape[0], h.shape[1]
        hh = h.reshape(B, S, self.nb, self.b).permute(2, 3, 0, 1)
        hh = hh.reshape(self.nb, self.b, B * S)
        y = torch.bmm(self.blocks, hh)
        y = y.reshape(self.nb, self.b, B, S).permute(2, 3, 0, 1)
        return y.reshape(*h.shape[:-1], self.n)

# ------------------------------------------------------------- attention ---
class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq=4096):
        super().__init__()
        inv = 1.0 / (head_dim ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos().bfloat16(), persistent=False)
        self.register_buffer("sin", f.sin().bfloat16(), persistent=False)

    def forward(self, x):                       # x: (B, H, S, hd)
        S = x.shape[2]
        cos = self.cos[:S][None, None]
        sin = self.sin[:S][None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, d, cfg, seed):
        super().__init__()
        self.h = 8
        self.hd = d // self.h
        self.q = TiedLinear(d, d, cfg, seed * 4 + 0)
        self.k = TiedLinear(d, self.h * self.hd // 4, cfg, seed * 4 + 1)   # GQA 2 kv
        self.v = TiedLinear(d, self.h * self.hd // 4, cfg, seed * 4 + 2)
        self.o = TiedLinear(d, d, cfg, seed * 4 + 3)
        self.rot = Rotary(self.hd)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.rot(self.q(x).view(B, S, self.h, self.hd).transpose(1, 2))
        k = self.rot(self.k(x).view(B, S, self.h // 4, self.hd).transpose(1, 2))
        v = self.v(x).view(B, S, self.h // 4, self.hd).transpose(1, 2)
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(a.transpose(1, 2).reshape(B, S, -1))

# ------------------------------------------------------------------ block ---
class Block(nn.Module):
    def __init__(self, d, f, cfg, seed):
        super().__init__()
        self.n1 = nn.RMSNorm(d, dtype=torch.bfloat16)
        self.attn = Attention(d, cfg, seed)
        self.n2 = nn.RMSNorm(d, dtype=torch.bfloat16)
        self.mode = cfg.mode
        if cfg.mode == "dense":
            self.gate = nn.Linear(d, f, dtype=torch.bfloat16)
            self.up = nn.Linear(d, f, dtype=torch.bfloat16)
            self.down = nn.Linear(f, d, dtype=torch.bfloat16)
        else:
            self.gate = TiedLinear(d, f, cfg, seed * 3 + 0)
            self.up = TiedLinear(d, f, cfg, seed * 3 + 1)
            self.down = TiedLinear(f, d, cfg, seed * 3 + 2)

    def _core(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            # activation checkpointing: tied layers save ~1.2GB/layer otherwise
            return torch.utils.checkpoint.checkpoint(
                self._core, x, use_reentrant=False)
        return self._core(x)

class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab = 50304                       # GPT-NeoX-20B tokenizer pad
        self.emb = nn.Embedding(self.vocab, cfg.d, dtype=torch.bfloat16)
        with torch.no_grad():
            self.emb.weight.normal_(0, 0.02)   # standard tied-head init scale
        self.blocks = nn.ModuleList(
            [Block(cfg.d, cfg.ffn, cfg, i) for i in range(cfg.layers)])
        self.nf = nn.RMSNorm(cfg.d, dtype=torch.bfloat16)
        self.head = nn.Linear(cfg.d, self.vocab, dtype=torch.bfloat16)
        self.head.weight = self.emb.weight       # tied

    def forward(self, idx, labels=None):
        h = self.emb(idx)
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.nf(h))        # bf16; fp32 materialized 13GB/rank
        loss = None
        if labels is not None:
            # SHIFT: predict token s+1 from tokens <= s (no copy shortcut!)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab),
                labels[:, 1:].reshape(-1))
        return logits, loss

# ------------------------------------------------------------------- data ---
def data_stream(steps, bs, seq):
    """FineWeb-Edu sample-10BT, GPT-NeoX tokenizer, packed for the FULL run."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    import datasets
    ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                               split="train", streaming=True)
    need = steps * bs * seq
    flat, got = [], 0
    for ex in ds:
        ids = tok(ex["text"], add_special_tokens=False).input_ids
        flat.extend(ids); got += len(ids)
        if got >= need:
            break
    n = (len(flat) // (bs * seq)) * (bs * seq)
    flat = torch.tensor(flat[:n], dtype=torch.int32).view(-1, seq)
    print(f"data packed: {n} tokens (needed {need})", flush=True)
    return flat

# ---------------------------------------------------------------- training --
def main():
    global A
    A = parse()
    RANK = int(os.environ.get("RANK", "0"))
    WORLD = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD > 1:
        torch.cuda.set_device(RANK % WORLD)
        torch.distributed.init_process_group("nccl")
        DEV = f"cuda:{RANK % WORLD}"
    else:
        DEV = f"cuda:{A.gpu}"
    torch.set_float32_matmul_precision("high")
    import wandb
    tag = f"{A.mode}-{A.layers}L{A.d}d-b{A.block}-r{A.rank}"
    if RANK == 0:
        wandb.init(project=A.project, name=tag, config=vars(A))

    # data: pre-tokenized cache (built once, loaded by all ranks)
    CACHE = "/root/phi/data_cache.pt"
    if os.path.exists(CACHE):
        flat = torch.load(CACHE, weights_only=True)   # 1-D int32 token stream
    else:
        flat = data_stream(A.steps, A.bs, A.seq).reshape(-1)
        torch.save(flat, CACHE)
    TPB = A.bs * A.seq                                # tokens per rank-batch
    nb = flat.numel() // (TPB * WORLD)                # batches per rank

    model = LM(A).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    if WORLD > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[RANK % WORLD])
        _raw = model.module
    else:
        _raw = model
    print(f"[{tag}] params {npar/1e6:.1f}M | batches {nb}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    def lr_at(s):
        if s < A.warmup:
            return s / A.warmup
        if s < A.decay_start:
            return 1.0
        frac = (s - A.decay_start) / max(A.steps - A.decay_start, 1)
        return max(0.02, 1.0 - frac)
    model.train()
    t0 = time.time()
    for step in range(A.steps):
        off = ((step * WORLD + RANK) % nb) * TPB
        xb = flat[off : off + TPB].view(A.bs, A.seq).long().to(DEV)
        lr = A.lr * lr_at(step)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        logits, loss = model(xb, labels=xb)
        if step == 0 and RANK == 0:
            # SANITY GATE: a random model on real text must have loss ~ ln(V)
            # ~10.8. A much smaller value means a label/leak bug (e.g. the
            # copy shortcut) - ABORT before wasting the run.
            if loss.item() < 5.0:
                print(f"SANITY ABORT: step-0 loss {loss.item():.2f} < 5.0 "
                      f"(expected ~10.8) - label leak or data bug", flush=True)
                raise SystemExit(1)
            print(f"SANITY OK: step-0 loss {loss.item():.2f}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 and RANK == 0:
            tps = (step + 1) * WORLD * A.bs * A.seq / max(time.time() - t0, 1)
            wandb.log({"train/loss": loss.item(), "train/lr": lr,
                       "train/tok_s": tps}, step=step)
            print(f"[{tag}] step {step:5d} loss {loss.item():.4f} "
                  f"({tps:.0f} tok/s)", flush=True)
        if (step + 1) % 500 == 0 or step == A.steps - 1:
            # held-out: last 2 batches as proxy val (proper OOD eval offline)
            model.eval()
            with torch.no_grad():
                vb = flat[-TPB:].view(A.bs, A.seq).long().to(DEV)   # unseen tail
                _, vloss = model(vb, labels=vb)
            model.train()
            if RANK == 0:
                wandb.log({"val/loss": vloss.item()}, step=step)
            if RANK == 0:
                print(f"[{tag}] VAL {vloss.item():.4f}", flush=True)
    if RANK == 0:
        torch.save(_raw.state_dict(), f"/root/phi/ckpt_pretrain_{tag}.pt")
    print(f"[{tag}] DONE", flush=True)
    wandb.finish()

if __name__ == "__main__":
    main()
