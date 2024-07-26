import re
from textwrap import dedent

import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .lm_modules import TransformerBlock
from .sparse_mlp import SparseMLP
from .tokenization.tokenization import dec, enc

DEFAULT_SPARSE_MLPS = {
    "T0": "mlp_map_test/M0_S-3_R1_P0",
    "T1": "mlp_map_test/M1_S-3_R1_P0",
    "T2": "mlp_map_test/M2_S-3_R1_P0",
    "T3": "mlp_map_test/M3_S-3_R1_P0",
    
    "M0": "mlp/M0_S-4_R1_P0",
    "M1": "mlp/M1_S-4_R1_P0",
    "M2": "mlp/M2_S-4_R1_P0",
    "M3": "mlp/M3_S-4_R1_P0",
    
    "A0": "attn/A0_S-1_R1_P0",
    "A1": "attn/A1_S-1_R1_P0",
    "A2": "attn/A2_S-3_R1_P0",
    "A3": "attn/A3_S-3_R1_P0",

    "Ra0": "res_pre_attn/Ra0_S-3_R1_P0",
    "Ra1": "res_pre_attn/Ra1_S-3_R1_P0",
    "Ra2": "res_pre_attn/Ra2_S-3_R1_P0",
    "Ra3": "res_pre_attn/Ra3_S-3_R1_P0",

    "Rm0": "res_pre_mlp/Rm0_S-3_R1_P0",
    "Rm1": "res_pre_mlp/Rm1_S-3_R1_P0",
    "Rm2": "res_pre_mlp/Rm2_S-3_R1_P0",
    "Rm3": "res_pre_mlp/Rm3_S-3_R1_P0",
}


def parse_mlp_tag(mlp_tag):
    defaults_tag_pat = re.compile(
        r"(?P<mlp_type>(T|M|Rm|Ra|A|Mo))(?P<layer>\d+)(\D(?P<feature_idx>\d+))?"
    )
    defaults_match = defaults_tag_pat.fullmatch(mlp_tag)
    
    file_tag_pat = re.compile(r'(?P<full_name>(?P<mlp_type>(T|Mo|M|A|Rm|Ra))(?P<layer>\d+)_S[-\d]+.{0,6}_P\d+)([^\d](?P<feature_idx>\d+))?')
    full_file_match = file_tag_pat.fullmatch(mlp_tag)

    if defaults_match:
        match_groups = defaults_match.groupdict()
        mlp_type, layer, feature_idx = (
            match_groups["mlp_type"],
            int(match_groups["layer"]),
            match_groups["feature_idx"]
        )
        
        
        feature_idx = None if feature_idx is None else int(feature_idx)

        assert mlp_type+str(layer) in DEFAULT_SPARSE_MLPS
        return DEFAULT_SPARSE_MLPS[mlp_type+str(layer)], mlp_type, layer, feature_idx
    elif full_file_match:
        # try interpreting the mlp_tag as a filename

        mlp_type_to_file = {
            'M': 'mlp',
            'Mo': 'mlp',
            'A': 'attn_test',
            'T': 'mlp_map_test',
            'Ra': 'res_pre_attn',
            'Rm': 'res_pre_mlp'
        }

        match_groups = full_file_match.groupdict()

        full_name, mlp_type, layer, feature_idx = match_groups['full_name'], match_groups['mlp_type'], int(match_groups['layer']), match_groups['feature_idx']
        file = mlp_type_to_file[mlp_type] + '/' + full_name

        feature_idx = None if feature_idx is None else int(feature_idx)

        return file, mlp_type, layer, feature_idx
    else:
        return False


class TinyModel(nn.Module):
    def __init__(
        self,
        d_model=768,
        n_layers=4,
        n_heads=16,
        max_seq_len=256,
        vocab_size=10_000,
        from_pretrained="tiny_model",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = nn.Parameter(
            1e-4 * torch.randn(self.embed.weight.shape, requires_grad=True)
        )
        self.pos_embed = nn.Parameter(
            1e-4 * torch.randn(1, max_seq_len, d_model, requires_grad=True)
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
            assert (
                from_pretrained is False
            ), "from_pretrained kwarg must be False or a string specifying model"

        # mlp tag list
        self._sparse_tags = []
        
        '''
        If using NNSight, you can't return NNsight wrapped modules via self.sparse_mlps
        unless self.envoy is set. E.G.
        
        lm = TinyModel()
        model = NNSight(lm)
        lm.nnsight_proxy = model
        '''
        self.nnsight_proxy = None
        

    @property
    def dtype(self):
        return self.embed.weight.dtype

    @property
    def device(self):
        return self.embed.weight.device
    
    @property
    def proxy(self):
        return self if self.nnsight_proxy is None else self.nnsight_proxy
    
    @property
    def sparse_mlps(self):
        res = {}
        for layer in range(4):
            for mlp_type in ['A', 'T', 'M', 'Mo', 'Ra', 'Rm']:
                mlp_tag = f"{mlp_type}{layer}"
                if mlp_tag in self._sparse_tags:
                    if mlp_type == 'Ra':
                        res[mlp_tag] = self.proxy.torso[layer].res_pre_attn_sae
                    elif mlp_type == 'A':
                        res[mlp_tag] = self.proxy.torso[layer].attn_sae
                    elif mlp_type == 'Rm':
                        res[mlp_tag] = self.proxy.torso[layer].res_pre_mlp_sae
                    elif mlp_type == 'T':
                        res[mlp_tag] = self.proxy.torso[layer].transcoder
                    elif mlp_type in {'M', 'Mo'}:
                        res[mlp_tag] = self.proxy.torso[layer].mlp_sae
                    else:
                        raise ValueError(f'mlp_tag `{mlp_tag}` not found.')
        return res

    def get_upstream(self, downstream_tag):
        assert downstream_tag in self.sparse_mlps, f'downstream_tag `{downstream_tag}` not found in self.sparse_mlps'
        res = {}
        for mlp_tag, mlp in self.sparse_mlps.items():
            if mlp_tag == downstream_tag:
                break
            else:
                res[mlp_tag] = mlp
        return res
    
    def get_downstream(self, upstream_tag):
        assert upstream_tag in self.sparse_mlps, f'upstream_tag `{upstream_tag}` not found in self.sparse_mlps'
        res = {}
        mlp_tags = list(self.sparse_mlps.keys())
        downstream_tags = mlp_tags[mlp_tags.index(upstream_tag)+1:]
        res = {ds_tag: self.sparse_mlps[ds_tag] for ds_tag in downstream_tags}
        return res
    

    def forward(self, tok_ids, return_idx=None, disable_flashattn=False):
        T = tok_ids.shape[-1]
        x = self.embed(tok_ids) + self.pos_embed[:, :T]

        for layer_idx, layer in enumerate(self.torso):
            if layer_idx == return_idx:
                return x
            x = layer(x, disable_flashattn=disable_flashattn)
        logits = self.lm_head(x)
        return F.log_softmax(logits, dim=-1)

    def generate(self, prompt, n_toks=50, temperature=0.8, break_on_end=True):
        assert temperature >= 0.0
        toks = enc(prompt, add_begin=True).to(self.lm_head.weight.device)

        for _ in range(n_toks):
            with torch.no_grad():
                logprobs = self.forward(toks)[0, -1]
                if temperature == 0:
                    next_tok = logprobs.argmax().item()
                else:
                    next_tok = dists.Categorical(
                        logits=logprobs * (1 / temperature)
                    ).sample()
            toks = torch.cat((toks, torch.tensor([[next_tok]]).to(toks.device)), dim=-1)
            if break_on_end and next_tok == enc("[END]").item():
                break
            if toks.shape[1] >= self.max_seq_len:
                break
        return dec(toks[:, 1:])[0]

    def get_sparse_act_fn(self, mlp_tag=None, mlp=None):
        '''
        Returns `get_sparse_mlp_acts`, which takes in tok_ids and returns sparse mlp activations. It optionally allows `indices`.
        '''
        assert not (mlp_tag is None and mlp is None)

        parse_output = parse_mlp_tag(mlp_tag)
        

        if parse_output is False:
            assert False, dedent(
                'Failed to parse mlp.'
            )
            # assert False, dedent(
            #     """
            #     [STUB]
            #     That\'s not a valid MLP tag. Here are some examples of MLP tags:
            #     M0, A2, Rm0, Ra1, Mo3
            #     They start with a string in [M, A, Rm, Ra, Mo]
            #     representing mlp map, attn out SAE, residual pre-mlp SAE, residual pre-attn SAE, and MLP out SAE respectively.
            #     and they end with a number representing the layer.

            #     You can also specify individual feature_idxs, e.g. lm['A2.100'](tok_ids) to get the activations of neuron 100.
            #     """
            # )
        else:
            file, mlp_type, layer, feature_idx = parse_output
            mlp_tag = mlp_type + str(layer)
            if mlp is None:
                sparse_mlp = SparseMLP.from_pretrained(file).to(device=self.device, dtype=self.dtype)
            else:
                sparse_mlp = mlp.to(device=self.device, dtype=self.dtype)
            # else:
            #     assert False, dedent(
            #         """
            #         mlp_tag {mlp_tag} not found in tiny_model.sparse_mlps or DEFAULT_SPARSE_MLPS

            #         [STUB]: unimplemented                 
            #         To add a sparse_mlp, do e.g.
            #         tiny_model.set_saes({
            #            \'M2\': SparseMLP.from_pretrained(\'mlp_map/M0_S-1_B0_P0\')
            #         })
                                     
            #         Available keys (of form {mlp_type}{layer}) are:
            #            M0..3 (for MLPs)
            #            A0..3 (for Attn out)
            #            Rm0..3 (for SAE on the residual before MLP)
            #            Ra0..3 (for SAE on the residual stream before attn)
            #            Mo0..3 (for SAE on MLP out)
                    
            #         See https://huggingface.co/noanabeshima/tiny_model/tree/main for available sparse MLPs.
            #         """
            #     )

            def get_sparse_mlp_acts(tok_ids, indices=feature_idx):
                x = self.forward(tok_ids, return_idx=layer)
                if mlp_type == "Ra":
                    return sparse_mlp.get_acts(x, indices=indices)
                attn_out = self.torso[layer].attn(x)
                if mlp_type == "A":
                    return sparse_mlp.get_acts(attn_out, indices=indices)
                x = attn_out + x
                if mlp_type in {"T", "Rm"}:
                    return sparse_mlp.get_acts(x, indices=indices)
                else:
                    assert mlp_type == "M", "mlp_type must be one of Ra, A, M, Rm, T"
                    mlp_out = self.torso[layer].mlp(x)
                    return sparse_mlp.get_acts(mlp_out, indices=indices)

            return get_sparse_mlp_acts
    
    def __getitem__(self, index):
        """
        To be used like:
        sparse_acts = lm['A0'](tok_ids, indices=[1,5,100])
        sparse_acts = lm['M1'](tok_ids, indices=slice(0,100))
        sparse_acts = lm['M3'](tok_ids, indices=0)

        or for single neurons

        sparse_acts = lm['M2N100'](tok_ids)
        sparse_acts = lm['M2.100'](tok_ids)
        """
        if isinstance(index, int):
            return self.torso[index]
        else:
            mlp_tag = index
            return self.get_sparse_act_fn(mlp_tag)

    def register_sparse(self, mlp_tag=None, mlp=None, include_error=True, detach_error=True, detach_pred=False):
        assert not (mlp_tag is None and mlp is None)
        
        
        # If mlp_tag is a list, interpret mlp_tag and mlp as being lists of mlps to add
        if isinstance(mlp_tag, list):
            assert all([isinstance(tag, str) for tag in mlp_tag]),\
                'if mlp_tag is a list, must be a list of strings (mlp tags)'
            assert mlp is None or len(mlp) == len(mlp_tag)
            
            if mlp is None:
                mlp = [None for _ in range(len(mlp_tag))]
            
            # make arguments plural to make clear that they're lists
            mlp_tags, provided_mlps = mlp_tag, mlp
            
            kwargs = dict(include_error=include_error, detach_error=detach_error, detach_pred=detach_pred)
            for tag, provided_mlp in zip(mlp_tags, provided_mlps):
                self.register_sparse(mlp_tag=tag, mlp=provided_mlp, **kwargs)
            
            return


        parse_output = parse_mlp_tag(mlp_tag)
        
        if parse_output is False:
            assert False, dedent(
                'Failed to parse mlp.'
            )
        else:
            file, mlp_type, layer, feature_idx = parse_output
            assert feature_idx is None
            mlp_tag = mlp_type + str(layer)
            if mlp is None:
                sparse_mlp = SparseMLP.from_pretrained(
                                                        file,
                                                        include_error=include_error,
                                                        detach_error=detach_error,
                                                        detach_pred=detach_pred
                                                      )
            else:
                sparse_mlp = mlp
            
            sparse_mlp = sparse_mlp.to(device=self.device, dtype=self.dtype)
            
            if mlp_type != "T":
                sparse_mlp.register_full_backward_hook(lambda m, grad_in, grad_out: (grad_out,))
            else:
                # [STUB] - warning that transcoders don't backward through transcoder errors.
                pass
            
            transformer_block = self.proxy.torso[layer]
            if mlp_type == 'Ra':
                transformer_block.res_pre_attn_sae = sparse_mlp
            elif mlp_type == "A":
                transformer_block.attn_sae = sparse_mlp
            elif mlp_type == 'Rm':
                transformer_block.res_pre_mlp_sae = sparse_mlp
            elif mlp_type == "T":
                transformer_block.transcoder = sparse_mlp
            elif mlp_type == 'M':
                transformer_block.mlp_sae = sparse_mlp
            
            
            else:
                raise ValueError(f'mlp_type {mlp_type} is unsupported.')
            
            
            self._sparse_tags.append(mlp_tag)
    
    def wipe_sparse(self):
        for block in self.proxy.torso:
            block.attn_sae = None
            block.transcoder = None
            block.mlp_sae = None
            
            


def get_state_dict(model_fname="tiny_model"):
    """
    There are two models available: `tiny_model` and `tiny_model_1_epoch`.
    """
    assert model_fname in [
        "tiny_model",
        "tiny_model_1_epoch",
    ], "There are two models available: `tiny_model` and `tiny_model_1_epoch`."
    state_dict = torch.load(
        hf_hub_download(repo_id="noanabeshima/tiny_model", filename=f"{model_fname}.pt"),
        map_location=torch.device('cpu')
    )
    return state_dict
