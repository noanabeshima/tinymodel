import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class SparseMLP(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.encoder = nn.Linear(d_model, n_features)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(n_features, d_model)

    def get_acts(self, x, indices=None):
        """Indices are either a slice, an int, or a list of ints"""
        if indices is None:
            return self.act(self.encoder(x))
        preacts = x @ self.encoder.weight.T[:, indices] + self.encoder.bias[indices]
        return self.act(preacts)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.act(x)
        x = self.decoder(x)
        return x

    @classmethod
    def from_pretrained(self, state_dict_path: str, repo_id="noanabeshima/tiny_model"):
        """Uses huggingface_hub to download an SAE/sparse MLP."""
        state_dict = torch.load(
            hf_hub_download(repo_id=repo_id, filename=state_dict_path + ".pt")
        )
        n_features, d_model = state_dict["encoder.weight"].shape
        mlp = SparseMLP(d_model=d_model, n_features=n_features)
        mlp.load_state_dict(state_dict)
        return mlp
