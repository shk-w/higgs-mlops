import importlib
import json

import pytest
from fastapi.testclient import TestClient

from higgs.model import HiggsMLP, ModelConfig, save_checkpoint

N_FEATURES = 28


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Build a tiny untrained model + identity scaler, so no real artifacts are needed
    art = tmp_path_factory.mktemp("artifacts")
    save_checkpoint(
        HiggsMLP(ModelConfig(in_features=N_FEATURES, hidden_dims=(8,))), art / "model.pt"
    )
    (art / "scaler.json").write_text(
        json.dumps({"mean": [0.0] * N_FEATURES, "std": [1.0] * N_FEATURES})
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HIGGS_ARTIFACTS", str(art))
        from higgs import serve

        importlib.reload(serve)  # re-run the module-level loading with the temp folder
        with TestClient(serve.app) as c:
            yield c


def test_valid_payload_returns_200(client):
    r = client.post("/predict", json={"features": [0.0] * N_FEATURES})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("signal", "background")
    assert 0.0 <= body["probability"] <= 1.0


def test_wrong_length_returns_422(client):
    r = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422
