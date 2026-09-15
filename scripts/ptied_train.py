import types
import torch
from pretrain_tb_fast import TiedLinear

def _slim_w_idx(tl):
    """w_idx for the [n2, d_out] slim Wt (only the used block-columns)."""
    nb_out, b = tl.nb_out, tl.b
    rows = torch.arange(nb_out, device=tl.w_idx.device)[:, None, None] * b \
        + torch.arange(b, device=tl.w_idx.device)[None, :, None]
    cols = torch.arange(nb_out, device=tl.w_idx.device)[:, None, None] * b \
        + torch.arange(b, device=tl.w_idx.device)[None, None, :]
    return (rows * tl.d_out + cols).reshape(-1)

def ptied_forward(self, x):
    """Training path: the TiedLinear as ONE dense operator, differentiable.

    P = (Mf @ Wt_slim) + V.T @ U.T, rebuilt each call (per optimizer step);
    forward and backward are plain dense GEMMs. Math identical to
    TiedLinear.forward up to bf16 reassociation.
    """
    if not hasattr(self, "_w_idx_slim"):
        self._w_idx_slim = _slim_w_idx(self)
    n2, d_out, nb_out = self.n2, self.d_out, self.nb_out
    B, S = x.shape[0], x.shape[1]
    Wt = torch.zeros(n2 * d_out, dtype=x.dtype, device=x.device)
    Wt = Wt.index_copy(0, self._w_idx_slim,
                       self.blocks[:nb_out].transpose(-1, -2).reshape(-1))
    P = self.Mf @ Wt.view(n2, d_out) + self.V.T @ self.U.T
    y = x.reshape(B * S, self.d_in) @ P
    return y.reshape(B, S, d_out)

def enable_ptied(model):
    """Swap every TiedLinear's forward to the merged-operator form.
    Modules and parameters stay intact (optimizer state unchanged)."""
    n = 0
    for mod in model.modules():
        for attr, child in list(mod.named_children()):
            if isinstance(child, TiedLinear):
                child.forward = types.MethodType(ptied_forward, child)
                n += 1
    return n

def disable_ptied(model):
    for mod in model.modules():
        for attr, child in list(mod.named_children()):
            if isinstance(child, TiedLinear) and \
                    isinstance(child.forward, types.MethodType) and \
                    child.forward.__func__ is ptied_forward:
                child.forward = types.MethodType(TiedLinear.forward, child)
    return None
