import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HookPoint(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x

    def __repr__(self):
        if self.name is not None:
            return f"HookPoint('{self.name}')"
        else:
            return "HookPoint()"


class Attention(nn.Module):
    def __init__(self, n_heads, d_model, d_head, max_seq_len):
        super().__init__()
        self.Q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.K = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.V = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.O = nn.Linear(d_head * n_heads, d_model)

        nn.init.normal_(self.Q.weight, std=np.sqrt(2 / (d_model + d_head)))
        nn.init.normal_(self.K.weight, std=np.sqrt(2 / (d_model + d_head)))
        nn.init.zeros_(self.O.bias)

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        self.attn_inp = HookPoint()
        self.qs = HookPoint()
        self.ks = HookPoint()
        self.vs = HookPoint()
        self.head_writeouts = HookPoint()
        self.catted_head_writeouts = HookPoint()
        self.attn_out = HookPoint()

    @property
    def Wq(self):
        return einops.rearrange(
            self.Q.weight.detach(), "d (h k) -> h d k", h=self.n_heads
        )

    @property
    def Wk(self):
        return einops.rearrange(
            self.K.weight.detach(), "d (h k) -> h d k", h=self.n_heads
        )

    @property
    def Wv(self):
        return einops.rearrange(
            self.V.weight.detach(), "d (h k) -> h d k", h=self.n_heads
        )

    @property
    def Wo(self):
        return self.O.weight.detach()

    def forward(self, x):
        x = self.attn_inp(x)  # hookpoint

        q, k, v = self.Q(x), self.K(x), self.V(x)

        qs = einops.rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        qs = self.qs(qs)  # hookpoint

        ks = einops.rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        ks = self.ks(ks)  # hookpoint

        vs = einops.rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)
        vs = self.vs(vs)  # hookpoint

        # force torch to use flash attention 2
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                head_writeouts = F.scaled_dot_product_attention(
                    qs, ks, vs, is_causal=True
                )
        else:
            head_writeouts = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        head_writeouts = self.head_writeouts(head_writeouts)  # hookpoint

        catted_head_writeouts = einops.rearrange(head_writeouts, "b h q d -> b q (h d)")
        catted_head_writeouts = self.catted_head_writeouts(
            catted_head_writeouts
        )  # hookpoint

        attn_out = self.O(catted_head_writeouts)
        attn_out = self.attn_out(attn_out)  # hookpoint

        return attn_out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.read_in = nn.Linear(d_model, d_mlp)
        self.act = nn.ReLU()
        self.write_out = nn.Linear(d_mlp, d_model)

        self.d_model = d_model
        self.d_mlp = d_mlp

        self.mlp_inp = HookPoint()
        self.mlp_out = HookPoint()

    def forward(self, x):
        x = self.mlp_inp(x)  # hookpoint

        preacts = self.read_in(x)
        acts = self.act(preacts)
        mlp_out = self.write_out(acts)

        mlp_out = self.mlp_out(mlp_out)  # hookpoint

        return mlp_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        d_head = d_model // n_heads

        self.attn = Attention(
            n_heads=n_heads, d_model=d_model, d_head=d_head, max_seq_len=max_seq_len
        )
        self.mlp = MLP(d_model=d_model, d_mlp=4 * d_model)

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.res_attn = HookPoint()
        self.res_mlp = HookPoint()
        self.res_final = HookPoint()

    def forward(self, x):
        x = self.res_attn(x)  # hookpoint
        
        attn_x = self.attn(x)
        x = attn_x + x

        x = self.res_mlp(x)  # hookpoint

        mlp_x = self.mlp(x)
        x = mlp_x + x

        x = self.res_final(x)  # hookpoint
        
        return x
