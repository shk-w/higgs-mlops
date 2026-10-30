import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from higgs.data import X_test_scaled, X_train_scaled, y_test, y_train
from higgs.model import HiggsMLP, ModelConfig, load_checkpoint, save_checkpoint

ARTIFACTS = Path("artifacts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--no-batch-norm", action="store_true")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    p.add_argument("--sched-patience", type=int, default=2, help="epochs before the LR is halved")
    p.add_argument(
        "--val-size", type=float, default=0.15, help="fraction of train held out for validation"
    )
    p.add_argument("--seed", type=int, default=10)
    return p.parse_args()


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool = False
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb.to(device))).cpu())
        labels.append(yb)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return roc_auc_score(labels, probs), accuracy_score(labels, probs >= 0.5)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data -----------------------------------------------------------
    # OpenML targets are categorical strings ('0'/'1'), so convert to ints
    y_tr = y_train.astype(str).astype(int).values
    y_te = y_test.astype(str).astype(int).values

    # Hold out part of the training set for early stopping, so the test set
    # is only used once at the very end
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_tr, test_size=args.val_size, random_state=args.seed, stratify=y_tr
    )

    train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True, drop_last=True)
    val_loader = make_loader(X_val, y_val, 1024, shuffle=False)
    test_loader = make_loader(X_test_scaled, y_te, 1024, shuffle=False)

    # --- Model ----------------------------------------------------------
    model = HiggsMLP(ModelConfig(in_features=X_tr.shape[1])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.sched_patience
    )

    # --- Training loop with early stopping on validation AUC ------------
    ARTIFACTS.mkdir(exist_ok=True)
    best_auc, stale, _last_epoch = 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)

        train_loss = running_loss / (len(train_loader) * args.batch_size)
        val_auc, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val AUC {val_auc:.4f} "
            f"| val acc {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            save_checkpoint(model, ARTIFACTS / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    # --- Final evaluation with the best checkpoint ----------------------
    best_model = load_checkpoint(ARTIFACTS / "model.pt", device)
    test_auc, test_acc = evaluate(best_model, test_loader, device)
    print(f"\nTest AUC {test_auc:.4f} | test acc {test_acc:.4f}")

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({"val_auc": best_auc, "test_auc": test_auc, "test_acc": test_acc}, f, indent=2)


if __name__ == "__main__":
    main()
