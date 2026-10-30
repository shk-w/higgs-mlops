import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from higgs.model import load_scaler

N_FEATURES = 28


def test_scaler_json_gives_zero_mean_unit_std(tmp_path):
    rng = np.random.default_rng(0)
    # features with very different offsets and scales
    X = rng.normal(5.0, 3.0, size=(5000, N_FEATURES)) * rng.uniform(0.1, 100, N_FEATURES)
    X = X.astype("float32")

    scaler = StandardScaler().fit(X)
    path = tmp_path / "scaler.json"
    path.write_text(json.dumps({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}))

    mean, std = load_scaler(path)  # the serving-side loader
    scaled = (torch.from_numpy(X) - mean) / std

    assert torch.allclose(scaled.mean(dim=0), torch.zeros(N_FEATURES), atol=1e-3)
    assert torch.allclose(scaled.std(dim=0, unbiased=False), torch.ones(N_FEATURES), atol=1e-3)
