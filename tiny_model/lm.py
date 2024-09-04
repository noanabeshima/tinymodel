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
    # "M0": "mlp_map_test/M0_S-6_R2_P2",
    # "M1": "mlp_map_test/M1_S-4_R8_P2",

    "M0": "mlp/M0_S-2_R1_P0",
    "M1": "mlp/M1_S-2_R1_P0",
    "M2": "mlp/M2_S-2_R1_P0",
    "M3": "mlp/M3_S-1_B0_P0",

    "A0": "attn/A0_S-2_B0_P1",
    "A1": "attn/A1_S-1_B0_P1",
    "A2": "attn/A2_S-2_B0_P1",
    "A3": "attn/A3_S-1_B2_P1",
}


def parse_mlp_tag(mlp_tag):
    defaults_tag_pat = re.compile(
        r"(?P<mlp_type>(M|Rm|Ra|A|Mo))(?P<layer>\d+)(\D(?P<feature_idx>\d+))?"
    )
    defaults_match = defaults_tag_pat.fullmatch(mlp_tag)
    file_tag_pat = re.compile(r'(?P<full_name>(?P<mlp_type>(Mo|M|A|Rm|Ra))(?P<layer>\d+)_S[-\d]+.{0,6}_P\d+)([^\d](?P<feature_idx>\d+))?')
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
            # 'Mo': 'mlp_out',
            'A': 'attn_test',
            'M': 'mlp_map_test',
            # 'Ra': 'res_pre_attn',
            # 'Rm': 'res_pre_mlp'
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

        # Dict from mlp_tag to sparse mlp
        self.sparse_mlps = nn.ModuleDict()

    @property
    def dtype(self):
        return self.embed.weight.dtype

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, tok_ids, return_idx=None):
        T = tok_ids.shape[-1]
        x = self.embed(tok_ids) + self.pos_embed[:, :T]
        if return_idx is not None:
            assert isinstance(return_idx, int)
            assert 0 <= return_idx and return_idx <= self.n_layers
            for layer_idx, layer in enumerate(self.torso):
                if layer_idx == return_idx:
                    return x
                x = layer(x)
        else:
            x = self.torso(x)
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

    def sparse_mlp(self, mlp_tag=None, mlp=None):
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
                if mlp_type in {"M", "Rm"}:
                    return sparse_mlp.get_acts(x, indices=indices)
                else:
                    assert mlp_type == "Mo", "mlp_type must be one of Ra, A, M, Rm, Mo"
                    mlp_out = self.torso[layer].mlp(x)
                    return sparse_mlp.get_acts(mlp_out, indices=indices)

            return get_sparse_mlp_acts
    
    def __getitem__(self, mlp_tag):
        """
        To be used like:
        sparse_acts = lm['A0'](tok_ids, indices=[1,5,100])
        sparse_acts = lm['M1'](tok_ids, indices=slice(0,100))
        sparse_acts = lm['M3'](tok_ids, indices=0)

        or for single neurons

        sparse_acts = lm['M2N100'](tok_ids)
        sparse_acts = lm['M2.100'](tok_ids)
        """
        return self.sparse_mlp(mlp_tag)


def get_state_dict(model_fname="tiny_model"):
    assert model_fname in [
        "tiny_model",
        "tiny_model_2L_1E",
        "tiny_model_2L_3E"
    ], "There are 3 models available: `tiny_model`, `tiny_model_2L_1E`, and `tiny_model_2L_3E`."
    state_dict = torch.load(
        hf_hub_download(repo_id="noanabeshima/tiny_model", filename=f"{model_fname}.pt"),
        map_location=torch.device('cpu')
    )
    return state_dict
