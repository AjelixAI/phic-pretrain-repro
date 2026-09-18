"""Widen the low-rank corrections of a tied-blocks checkpoint, zero-padded.
Function-preserving: the extra rank columns are exactly zero, so
V @ U.T is unchanged. Optimizer moments padded with zeros (fresh dims).
Usage: widen_rank.py SRC DST RANK"""
import sys, torch
sys.path.insert(0, '/root/phi')
src, dst, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(src, map_location='cpu', weights_only=True)

def widen_state(sd, R):
    # U/V tensors: rank dim is the LAST dim (=64); opt state nests tensors in dicts
    if isinstance(sd, torch.Tensor):
        return sd, 0
    out = {}
    n_widened = 0
    for k, v in sd.items():
        if isinstance(v, dict):
            w, n = widen_state(v, R); out[k] = w; n_widened += n
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, torch.Tensor) and v.dtype.is_floating_point and v.dim() >= 2 and 64 in v.shape and ('.U' in k or k.endswith('.U') or '.V' in k or k.endswith('.V')):
            if '.U' in k or k.endswith('.U'):      # U: [d_in, rank] -> pad last dim
                new = torch.zeros(*v.shape[:-1], R, dtype=v.dtype); new[..., :64] = v
            else:                                   # V: [rank, d_out] -> pad first dim
                new = torch.zeros(R, *v.shape[1:], dtype=v.dtype); new[:64] = v
            out[k] = new; n_widened += 1
        else:
            out[k] = v
    return out, n_widened

sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
w, n = widen_state(sd, R)
ck['model'] = w
if isinstance(ck.get('opt'), dict) and isinstance(ck['opt'].get('state'), dict):
    # opt state is keyed by param index; find the U/V indices via a rank-64 model
    from types import SimpleNamespace as NS
    import pretrain_tb_fast as pt
    a = ck['args']
    cfg = NS(**{k: v for k, v in a.items()}) if isinstance(a, dict) else a
    cfg.rank = 64
    m0 = pt.LM(cfg)
    names = [n for n, _ in m0.named_parameters()]
    idx_uv = [i for i, n in enumerate(names) if n.endswith('.U') or n.endswith('.V')]
    st = ck['opt']['state']
    n2 = 0
    for idx in idx_uv:
        for kk in ('exp_avg', 'exp_avg_sq'):
            t = st.get(idx, {}).get(kk)
            if t is None: continue
            if '.U' in names[idx] or names[idx].endswith('.U'):
                new = torch.zeros(*t.shape[:-1], R, dtype=t.dtype); new[..., :64] = t
            else:
                new = torch.zeros(R, *t.shape[1:], dtype=t.dtype); new[:64] = t
            st[idx][kk] = new; n2 += 1
    print(f"widened model tensors: {n}, opt tensors: {n2} (of {len(idx_uv)} U/V params x2)")
else:
    print(f"widened model tensors: {n} (no opt state in ckpt)")
ck['args']['rank'] = R if isinstance(ck.get('args'), dict) else None
torch.save(ck, dst)
print(f"saved {dst}")
