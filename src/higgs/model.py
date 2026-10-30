import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

NUM_FEATURES = 28


@dataclass
class ModelConfig:
    in_features: int = NUM_FEATURES
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    batch_norm: bool = True


class HiggsMLP(nn.Module):
    """Linear -> ReLU -> Dropout, repeated, then a single-logit output layer."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        layers: list[nn.Module] = []
        dim = c.in_features
        for h in c.hidden_dims:
            layers.append(nn.Linear(dim, h))
            if c.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(c.dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch,)."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))


def save_checkpoint(model: HiggsMLP, path: str | Path = "artifacts/model.pt") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.config)
    cfg["hidden_dims"] = list(cfg["hidden_dims"])
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path = "artifacts/model.pt", device="cpu") -> HiggsMLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
    model = HiggsMLP(ModelConfig(**cfg))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_scaler(path: str | Path = "artifacts/scaler.json"):
    """Returns (mean, std) tensors from the JSON saved at training time."""
    with open(path) as f:
        d = json.load(f)
    mean = torch.tensor(d["mean"], dtype=torch.float32)
    std = torch.tensor(d["std"], dtype=torch.float32)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


if __name__ == "__main__":
    model = HiggsMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")  # ~12k
    print("Logits shape:", model(torch.randn(8, NUM_FEATURES)).shape)
