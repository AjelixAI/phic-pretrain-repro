#!/usr/bin/env python3
"""
pretrain_tb.py — from-scratch pretraining head-to-head:
  dense baseline vs tied-blocks architecture, OLMo-style recipe,
  FineWeb-Edu open data, W&B-logged. One process per GPU.

Usage:
  python pretrain_tb.py --mode dense --gpu 0 --steps 6000
  python pretrain_tb.py --mode tied  --gpu 1 --steps 6000
"""
import numpy as np
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
    p.add_argument("--cache", default="/root/phi/data_cache.pt")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--chunked-ce", type=int, default=0,
                   help="rows per CE chunk; 0 = materialized logits (old path)")
    p.add_argument("--no-ckpt", action="store_true",
                   help="disable activation checkpointing (fits now with fast path)")
    p.add_argument("--init-from", default=None,
                   help="mid-training: load weights from a checkpoint (fresh optimizer, OLMo reset_trainer_state pattern)")
    p.add_argument("--no-ptied", action="store_true",
                   help="disable the P-form merged-operator training path")
    p.add_argument("--tag-suffix", default="",
                   help="appended to saved checkpoint names (protects prior artifacts)")
    p.add_argument("--cache2", default=None,
                   help="second data cache; training reads it from --switch-step on (staged anneal)")
    p.add_argument("--switch-step", type=int, default=0,
                   help="step at which the data source switches to --cache2")
    p.add_argument("--force-save", action="store_true",
                   help="allow overwriting an existing checkpoint file (default: refuse)")
    p.add_argument("--save-every", type=int, default=2500,
                   help="full-checkpoint interval (weights+optimizer+step); 0 = final only")
    p.add_argument("--resume", default=None,
                   help="full checkpoint (weights+optimizer+step) to resume from")
    p.add_argument("--rope-base", type=float, default=10000.0,
                   help="RoPE theta base; default 10000 (llama standard). 64 reproduces the old buggy encoding")
    p.add_argument("--vocab", type=int, default=50304,
                   help="tokenizer vocab size (50304 = GPT-NeoX; 100352 = OLMo 2 tokenizer)")
    p.add_argument("--val-file", default=None,
                   help="held-out general validation slice (disjoint from the train cache)")
    p.add_argument("--val-file2", default=None,
                   help="held-out anneal-domain validation slice (the skill-gain metric)")
    p.add_argument("--graphs", action="store_true",
                   help="capture the training step as ONE CUDA graph (the math-identical execution, the dispatch removed)")
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
    """Fast path: the frozen butterfly basis composes to ONE dense operator
    (non-persistent buffer, regenerated from seed at init - the file/ftory
    story is unchanged). Trained `blocks` + rank-r residual: identical math.

    Function is exactly the old TiedLinear up to bf16 reassociation order.
    """
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
        # --- compose the frozen basis into one dense operator --------------
        # basis(I)[0][j, i] = Op[i, j]  =>  y_flat = x_flat @ Mout
        with torch.no_grad():
            eye = torch.eye(n, dtype=torch.bfloat16)
            Mout = self.basis(eye.unsqueeze(0))[0]          # [n, n]
            nb_out = (d_out + self.b - 1) // self.b
            self.nb_out = nb_out
            self.register_buffer(
                "Mf", Mout[: self.d_in, : nb_out * self.b].contiguous(),
                persistent=False)
            n2 = nb_out * self.b
            rows = torch.arange(nb_out, device="cpu")[:, None, None] * self.b \
                + torch.arange(self.b, device="cpu")[None, :, None]
            cols = torch.arange(nb_out, device="cpu")[:, None, None] * self.b \
                + torch.arange(self.b, device="cpu")[None, None, :]
            self.register_buffer(
                "w_idx", (rows * n2 + cols).reshape(-1), persistent=False)
            self.n2 = n2

    def _bd(self, h):
        # trained block-diagonal multiply via ONE dense GEMM:
        # W_total = blockdiag(blocks[:nb_out]) built by a single precomputed-
        # index scatter (identical function to the per-block bmm, bf16
        # reassociation order differs only)
        B, S = h.shape[0], h.shape[1]
        n2 = self.n2
        Wt = torch.zeros(n2 * n2, dtype=h.dtype, device=h.device)
        Wt = Wt.index_copy(0, self.w_idx,
                           self.blocks[: self.nb_out].transpose(-1, -2)
                           .reshape(-1))
        y = h.reshape(B * S, n2) @ Wt.view(n2, n2)
        return y.reshape(B, S, n2)

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        h = x.reshape(B * S, self.d_in) @ self.Mf           # one GEMM
        h = h.reshape(B, S, self.nb_out * self.b)
        h = self._bd(h)
        y = h[..., : self.d_out] + x @ self.V.T @ self.U.T
        return y

# ------------------------------------------------------------- attention ---
class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq=4096, base=10000):
        super().__init__()
        # BUG FIX: was head_dim ** (...) — the head_dim used as the RoPE base,
        # aliasing the positional signal 6x within the 2048 context
        # (slowest wavelength 353 << 2048). Standard llama theta = 10000.
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
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
        self.rot = Rotary(self.hd, base=getattr(cfg, "rope_base", 10000.0))

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
        if self.training and torch.is_grad_enabled() and not getattr(A, "no_ckpt", False):
            # activation checkpointing: tied layers save ~1.2GB/layer otherwise
            return torch.utils.checkpoint.checkpoint(
                self._core, x, use_reentrant=False)
        return self._core(x)

class LM(nn.Module):
    def __init__(self, cfg):
        V = getattr(cfg, "vocab", 50304)
        V = getattr(cfg, "vocab", 50304)
        super().__init__()
        self.vocab = V                           # tokenizer vocab (cfg.vocab, default 50304)
        self.emb = nn.Embedding(self.vocab, cfg.d, dtype=torch.bfloat16)
        with torch.no_grad():
            self.emb.weight.normal_(0, 0.02)   # standard tied-head init scale
        self.blocks = nn.ModuleList(
            [Block(cfg.d, cfg.ffn, cfg, i) for i in range(cfg.layers)])
        self.nf = nn.RMSNorm(cfg.d, dtype=torch.bfloat16)
        self.head = nn.Linear(cfg.d, self.vocab, dtype=torch.bfloat16)
        self.head.weight = self.emb.weight       # tied
        self.chunk_rows = getattr(cfg, "chunked_ce", 0)

    def _ce_chunk(self, hc, yc):
        lg = self.head(hc)
        return F.cross_entropy(lg.float(), yc, reduction="sum")

    def forward(self, idx, labels=None):
        h = self.emb(idx)
        for b in self.blocks:
            h = b(h)
        hn = self.nf(h)
        if self.chunk_rows and labels is not None:
            # chunked CE: never materialize full-vocab logits; each chunk's
            # logits are recomputed in backward via activation checkpointing
            from torch.utils.checkpoint import checkpoint
            hf = hn[:, :-1].reshape(-1, hn.shape[-1])
            y = labels[:, 1:].reshape(-1)
            total = hf.shape[0]
            acc = None
            for i in range(0, total, self.chunk_rows):
                part = checkpoint(self._ce_chunk, hf[i:i + self.chunk_rows],
                                  y[i:i + self.chunk_rows], use_reentrant=False)
                acc = part if acc is None else acc + part
            return None, acc / total
        if labels is not None:
            logits = self.head(hn)            # bf16; fp32 materialized 13GB/rank
            # SHIFT: predict token s+1 from tokens <= s (no copy shortcut!)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab),
                labels[:, 1:].reshape(-1))
            return None, loss                  # logits unused in training/val
        return self.head(hn), None

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
def save_full(path, model, opt, step, A):
    """weights + optimizer state + step: the full resume checkpoint."""
    if os.path.exists(path) and not A.force_save and not path.endswith("running.pt"):
        raise SystemExit(f"SAVE REFUSED: {path} exists (use --force-save)")
    torch.save({"step": step, "model": model.state_dict(),
                "opt": opt.state_dict(), "args": vars(A)}, path)
    print(f"FULL SAVE {path} @step {step}", flush=True)

def main():
    global A
    A = parse()
    if os.environ.get("COMPILED_AUTOGRAD"):
        import torch._dynamo as _dynamo
        _dynamo.config.compiled_autograd = True
        print("compiled autograd: ON", flush=True)
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
    CACHE = A.cache
    if os.path.exists(CACHE):
        flat = torch.load(CACHE, weights_only=True, mmap=True)  # 1-D int32 stream (mmapped: shared across ranks)
    else:
        flat = data_stream(A.steps, A.bs, A.seq).reshape(-1)
        torch.save(flat, CACHE)
    TPB = A.bs * A.seq                                # tokens per rank-batch
    nb = flat.numel() // (TPB * WORLD)                # batches per rank
    if A.cache2:
        flat2 = torch.load(A.cache2, weights_only=True, mmap=True)
        nb2 = flat2.numel() // (TPB * WORLD)
        if RANK == 0:
            print(f"cache2 {A.cache2}: {flat2.numel()/1e9:.2f}B tok, "
                  f"switch at step {A.switch_step}", flush=True)

    model = LM(A).to(DEV)
    if A.init_from:
        sd = torch.load(A.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        del sd
        if RANK == 0:
            print(f"RESUMED weights from {A.init_from}", flush=True)
    ck_opt = None
    if A.resume:
        ck = torch.load(A.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        ck_opt = ck["opt"]
        A._start_step = ck["step"]
        del ck
        if RANK == 0:
            print(f"FULL RESUME from {A.resume} @step {A._start_step}",
                  flush=True)
    if not A.no_ptied:
        from ptied_train import enable_ptied
        n_swapped = enable_ptied(model)
        if RANK == 0:
            print(f"P-form training path ON ({n_swapped} TiedLinears)",
                  flush=True)
    if A.compile:
        import torch._dynamo as _dynamo_mod
        _dynamo_mod.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model, mode=os.environ.get("TCOMPILE_MODE", "default"))
    _graph_mode = getattr(A, "graphs", False)
    npar = sum(p.numel() for p in model.parameters())
    if WORLD > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[RANK % WORLD],
                    static_graph=_graph_mode)
        _raw = model.module
    else:
        _raw = model
    print(f"[{tag}] params {npar/1e6:.1f}M | batches {nb}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True,
                            capturable=_graph_mode)
    if ck_opt is not None:
        opt.load_state_dict(ck_opt)
        del ck_opt
        if RANK == 0:
            print("optimizer state restored", flush=True)

    def lr_at(s):
        if s < A.warmup:
            return s / A.warmup
        if s < A.decay_start:
            return 1.0
        frac = (s - A.decay_start) / max(A.steps - A.decay_start, 1)
        return max(0.02, 1.0 - frac)
    model.train()
    _graph = None
    xb_s = torch.zeros(A.bs, A.seq, dtype=torch.long, device=DEV)
    loss_s = torch.zeros((), device=DEV)
    opt_lr = torch.tensor(A.lr, device=DEV)
    t0 = time.time()
    _start = getattr(A, "_start_step", 0)
    for step in range(_start, A.steps):
        if A.cache2 and step >= A.switch_step:
            src, nb_src = flat2, nb2
        else:
            src, nb_src = flat, nb
        off = ((step * WORLD + RANK) % nb_src) * TPB
        xb = src[off : off + TPB].view(A.bs, A.seq).long().to(DEV)
        lr = A.lr * lr_at(step)
        if _graph_mode:
            opt_lr.fill_(lr)
            for gparam in opt.param_groups:
                gparam["lr"] = opt_lr
        else:
            for gparam in opt.param_groups:
                gparam["lr"] = lr
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        if _graph_mode and step - _start == 3:
            _graph = torch.cuda.CUDAGraph()
            xb_s.copy_(xb)
            with torch.cuda.graph(_graph):
                _, loss_s = model(xb_s, labels=xb_s)
                loss_s.backward()
                opt.step()
                opt.zero_grad(set_to_none=False)
            if RANK == 0:
                print(f"[{tag}] CUDA graph captured at step {step}", flush=True)
        if _graph_mode and step - _start > 3:
            xb_s.copy_(xb)
            _graph.replay()
            loss = loss_s
        else:
            _, loss = model(xb, labels=xb)
        if os.environ.get("COMPILED_AUTOGRAD"):
            loss = loss.clone()   # detach from cudagraph buffer before next replay
        if step == 0 and RANK == 0:
            import math
            # SANITY GATE: a random model on real text must have loss ~ ln(V)
            # ~10.8. A much smaller value means a label/leak bug (e.g. the
            # copy shortcut) - ABORT before wasting the run.
            if A.init_from or A.resume:
                # resume gate: step-0 loss must sit near the source
                # checkpoint's recorded plateau val (Phase-B stable plateau
                # ~4.125 nats); a big deviation = wrong ckpt/architecture
                if not (2.8 < loss.item() < 5.4):
                    print(f"RESUME ABORT: step-0 loss {loss.item():.2f} "
                          f"outside resume band [2.8, 5.4] vs plateau ~4.125",
                          flush=True)
                    raise SystemExit(1)
                print(f"RESUME GATE: step-0 loss {loss.item():.3f} vs "
                      f"recorded plateau ~4.125 -> PASS", flush=True)
            elif loss.item() < math.log(getattr(A, "vocab", 50304)) - 5.0:
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
            # held-out: --val-file (the held-out generic slice, disjoint from
            # the train cache by construction) or the cache tail as the proxy
            model.eval()
            with torch.no_grad():
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                if getattr(A, "val_file", None):
                    vf = torch.load(A.val_file, weights_only=True, mmap=True)
                    voff = (step * 7919) % max(vf.numel() - TPB, 1)
                    vb = torch.as_tensor(np.asarray(vf[voff:voff + TPB])).long().to(DEV).view(A.bs, A.seq)
                    _, vloss = model(vb, labels=vb)
                else:
                    vb = flat[-TPB:].view(A.bs, A.seq).long().to(DEV)   # unseen tail
                    _, vloss = model(vb, labels=vb)
                if getattr(A, "val_file2", None):
                    vf2 = torch.load(A.val_file2, weights_only=True, mmap=True)
                    voff2 = (step * 104729) % max(vf2.numel() - TPB, 1)
                    vb2 = torch.as_tensor(np.asarray(vf2[voff2:voff2 + TPB])).long().to(DEV).view(A.bs, A.seq)
                    _, vloss2 = model(vb2, labels=vb2)
            model.train()
            if RANK == 0:
                wandb.log({"val/general": vloss.item()}, step=step)
                if getattr(A, "val_file2", None):
                    wandb.log({"val/anneal_domain": vloss2.item()}, step=step)
            if RANK == 0:
                print(f"[{tag}] VAL general {vloss.item():.4f}"
                      + (f" | anneal {vloss2.item():.4f}" if getattr(A, "val_file2", None) else ""),
                      flush=True)
                if A.save_every and (step + 1 - _start) % A.save_every == 0:
                    save_full(f"/root/phi/ckpt_{tag}{A.tag_suffix}_running.pt",
                              _raw, opt, step + 1, A)
                    if (step + 1 - _start) % (A.save_every * 4) == 0:
                        save_full(f"/root/phi/ckpt_{tag}{A.tag_suffix}_mile_{step + 1}.pt",
                                  _raw, opt, step + 1, A)
                        miles = sorted(f for f in os.listdir('/root/phi')
                                       if f.startswith(f"ckpt_{tag}{A.tag_suffix}_mile_"))
                        for old in miles[:-5]:
                            os.remove('/root/phi/' + old)
                            print(f"milestone rotation: removed {old}", flush=True)
    if RANK == 0:
        outp = f"/root/phi/ckpt_pretrain_{tag}{A.tag_suffix}.pt"
        if os.path.exists(outp) and not A.force_save:
            raise SystemExit(f"SAVE REFUSED: {outp} exists (use --force-save)")
        torch.save(_raw.state_dict(), outp)
    print(f"[{tag}] DONE", flush=True)
    wandb.finish()

if __name__ == "__main__":
    main()
