import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class SparseModelError(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return (target - pred.detach())


class SparseMLP(nn.Module):
    def __init__(self, d_model, n_features, include_error=False, detach_error=False, detach_pred=False):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.encoder = nn.Linear(d_model, n_features)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(n_features, d_model)

        self.include_error = include_error
        self.detach_error = detach_error
        self.detach_pred = detach_pred

        if self.include_error:
            self.get_eps = SparseModelError()        

    def get_acts(self, x, indices=None):
        """Indices are either a slice, an int, or a list of ints"""
        if indices is None:
            return self.act(self.encoder(x))
        preacts = x @ self.encoder.weight.T[:, indices] + self.encoder.bias[indices]
        return self.act(preacts)

    def __call__(self, x, target=None):
        preacts = self.encoder(x)
        acts = self.act(preacts)
        pred = self.decoder(acts)
        
        if self.detach_pred:
            pred = pred.detach()
        
        if self.include_error:
            error = self.get_eps(pred, target)
            
            if self.detach_error:
                error = error.detach()
            
            return pred + error
        else:
            return pred

    @classmethod
    def from_pretrained(self, state_dict_path: str, repo_id="noanabeshima/tiny_model", include_error=True, detach_error=True, detach_pred=False, **kwargs):
        """Uses huggingface_hub to download an SAE/sparse MLP."""
        state_dict = torch.load(
            hf_hub_download(repo_id=repo_id, filename=state_dict_path + ".pt"),
            weights_only=True
        )
        n_features, d_model = state_dict["encoder.weight"].shape
    
        if 'n_features' in kwargs:
            assert kwargs['n_features'] == n_features
        if 'd_model' in kwargs:
            assert kwargs['d_model'] == d_model
        
        mlp = SparseMLP(d_model=d_model, n_features=n_features, include_error=include_error, detach_error=detach_error, detach_pred=detach_pred)
        mlp.load_state_dict(state_dict)
        return mlp

def get_sliced_mlp(mlp: SparseMLP, start: int, end: int):
    assert isinstance(start, int) and isinstance(end, int), (start, end)
    assert end > start
    assert end <= mlp.n_features and start >= 0
    new_mlp = SparseMLP(d_model=mlp.d_model, n_features=end-start, include_error=mlp.include_error, detach_error=mlp.detach_error, detach_pred=mlp.detach_pred)
    new_mlp.encoder.weight.data = mlp.encoder.weight.data[start:end]
    new_mlp.encoder.bias.data = mlp.encoder.bias.data[start:end]
    new_mlp.decoder.weight.data = mlp.decoder.weight.data[:,start:end]
    new_mlp.decoder.bias.data = mlp.decoder.bias.data

    return new_mlp

from typing import Iterable

def get_masked_mlp(mlp: SparseMLP, mask: list[int]):
    assert len(mask) > 0
    assert isinstance(mask, Iterable)
    assert all([isinstance(it, int) for it in mask])

    new_mlp = SparseMLP(d_model=mlp.d_model, n_features=len(mask), include_error=mlp.include_error, detach_error=mlp.detach_error, detach_pred=mlp.detach_pred)
    new_mlp.encoder.weight.data = mlp.encoder.weight.data[mask]
    new_mlp.encoder.bias.data = mlp.encoder.bias.data[mask]
    new_mlp.decoder.weight.data = mlp.decoder.weight.data[:,mask]
    new_mlp.decoder.bias.data = mlp.decoder.bias.data

    return new_mlp
