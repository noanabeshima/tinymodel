import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import os
import torch.distributions as dists
from tiny_model.tokenization import enc, dec
from huggingface_hub import hf_hub_download


current_file = os.path.abspath(__file__)
current_dir = '/'.join(current_file.split('/')[:-1])

def recursively_name_modules(module):
    for name, child in module.named_children():
        child.name = f"{module.name}.{name}" if hasattr(module, 'name') else name
        recursively_name_modules(child)

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
            return 'HookPoint()'


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
        return einops.rearrange(self.Q.weight.detach(), "d (h k) -> h d k", h=self.n_heads)
    @property
    def Wk(self):
        return einops.rearrange(self.K.weight.detach(), "d (h k) -> h d k", h=self.n_heads)
    @property
    def Wv(self):
        return einops.rearrange(self.V.weight.detach(), "d (h k) -> h d k", h=self.n_heads)

    @property
    def Wo(self):
        return self.O.weight.detach()

    def forward(self, x):
        x = self.attn_inp(x) #hookpoint

        q, k, v = self.Q(x), self.K(x), self.V(x)

        qs = einops.rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        qs = self.qs(qs) # hookpoint

        ks = einops.rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        ks = self.ks(ks) # hookpoint

        vs = einops.rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)
        vs = self.vs(vs) # hookpoint

        # force torch to use flash attention 2
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                head_writeouts = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        else:
            head_writeouts = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        head_writeouts = self.head_writeouts(head_writeouts) #hookpoint

        catted_head_writeouts = einops.rearrange(head_writeouts, "b h q d -> b q (h d)")
        catted_head_writeouts = self.catted_head_writeouts(catted_head_writeouts) #hookpoint

        attn_out = self.O(catted_head_writeouts)
        attn_out = self.attn_out(attn_out) #hookpoint

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
        x = self.mlp_inp(x) #hookpoint

        preacts = self.read_in(x)
        acts = self.act(preacts)
        mlp_out = self.write_out(acts)

        mlp_out = self.mlp_out(mlp_out) #hookpoint

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
        x = self.res_attn(x) # hookpoint
        assert x is not None
        x = self.attn(x) + x
        assert x is not None
        x = self.res_mlp(x) # hookpoint
        assert x is not None
        mlp_x = self.mlp(x)
        assert mlp_x is not None
        x = self.mlp(x) + x
        assert x is not None
        x = self.res_final(x) # hookpoint
        assert x is not None
        return x


class TinyModel(nn.Module):
    def __init__(self, d_model=768, n_layers=4, n_heads=16, max_seq_len=256, vocab_size=10_000, from_pretrained='tiny_model'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = nn.Parameter(
            .0001*(2*(torch.rand(self.embed.weight.shape, requires_grad=True)-0.5))
        )
        self.pos_embed = nn.Parameter(
            .0001*(2*(torch.rand(1, max_seq_len, d_model, requires_grad=True)-0.5))
        )

        self.torso = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        if isinstance(from_pretrained, str):
            self.load_state_dict(get_state_dict(from_pretrained))
        else:
            assert from_pretrained is False, 'from_pretrained kwarg must be False or a string specifying model'

    @property
    def dtype(self):
        return self.embed.weight.dtype

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, tok_ids, final_layer=None):
        T = tok_ids.shape[-1]
        x = self.embed(tok_ids) + self.pos_embed[:,:T]
        if final_layer is not None:
            assert isinstance(final_layer, int)
            assert 0 <= final_layer and final_layer < self.n_layers
            for layer_idx, layer in enumerate(self.torso):
                x = layer(x)
                if layer_idx == final_layer:
                    return x
        else:
            x = self.torso(x)
            logits = self.lm_head(x)
            return F.log_softmax(logits, dim=-1)

    def generate(self, prompt, n_toks=50, temperature=0.8, break_on_end=True):
        assert temperature >= 0.0
        toks = enc(prompt, add_begin=True).to(self.lm_head.weight.device)

        for _ in range(n_toks):
            with torch.no_grad():
                logprobs = self.forward(toks)[0,-1]
                if temperature == 0:
                    next_tok  = logprobs.argmax().item()
                else:
                    next_tok = dists.Categorical(logits=logprobs*(1/temperature)).sample()
            toks = torch.cat((toks, torch.tensor([[next_tok]]).to(toks.device)), dim=-1)
            if break_on_end and next_tok == enc('[END]').item():
                break
            if toks.shape[1] >= self.max_seq_len:
                break
        return dec(toks[:,1:])[0]
    
    def __getitem__(self, index):
        return self.torso[index]
            

def get_state_dict(model_fname='tiny_model'):
    '''
    There are two models available: `tiny_model` and `tiny_model_1_epoch`.
    '''
    assert model_fname in ['tiny_model', 'tiny_model_1_epoch'], 'There are two models available: `tiny_model` and `tiny_model_1_epoch`.'
    file_loc = hf_hub_download(repo_id="noanabeshima/tiny_model", filename=f"{model_fname}.pt")
    state_dict = torch.load(file_loc)
    return state_dict