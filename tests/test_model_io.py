import torch

from higgs.model import HiggsMLP, ModelConfig, load_checkpoint, save_checkpoint

N_FEATURES = 28


def test_forward_shape_and_probability_range():
    torch.manual_seed(0)
    model = HiggsMLP(ModelConfig(in_features=N_FEATURES))
    x = torch.randn(16, N_FEATURES)

    logits = model(x)
    probs = model.predict_proba(x)  # sigmoid applied

    assert logits.shape == (16,)
    assert probs.shape == (16,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_save_then_load_gives_identical_outputs(tmp_path):
    torch.manual_seed(0)
    model = HiggsMLP(ModelConfig(in_features=N_FEATURES, hidden_dims=(32, 16)))
    model.eval()  # BatchNorm and Dropout must be in eval mode to compare
    x = torch.randn(8, N_FEATURES)

    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)

    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
