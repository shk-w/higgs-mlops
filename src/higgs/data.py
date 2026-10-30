import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARFF_PATH = "src\\higgs\\dataset.arff"


def read_arff_header(path):
    """Return (column names, nominal column names, number of header lines)."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    names, nominal, n_lines = [], [], 0
    with opener(path, "rt") as f:
        for line in f:
            n_lines += 1
            low = line.strip().lower()
            if low.startswith("@attribute"):
                m = re.match(
                    r"@attribute\s+('(?:[^']|\\')*'|\"[^\"]*\"|\S+)\s+(.*)",
                    line.strip(),
                    re.IGNORECASE,
                )
                name = m.group(1).strip("'\"")
                names.append(name)
                if m.group(2).strip().startswith("{"):
                    nominal.append(name)
            elif low.startswith("@data"):
                break
    return names, nominal, n_lines


def load_data():
    """Read the ARFF file, drop rows with NaNs, return float32 features and 0/1 labels."""
    names, nominal, n_header = read_arff_header(ARFF_PATH)
    print(f"{len(names)} columns, nominal: {nominal}")

    df = pd.read_csv(
        ARFF_PATH,
        skiprows=n_header,
        header=None,
        names=names,
        na_values="?",
        dtype="float32",
    )

    n_before = len(df)
    df = df.dropna()
    print(f"Dropping {n_before - len(df)} of {n_before} rows with NaNs")

    target = "class" if "class" in df.columns else nominal[-1]
    y = df.pop(target).astype(int)
    print(f"Target column: '{target}', positive rate: {y.mean():.3f}")
    return df, y


def preprocess(train_data, test_data):

    train_scaled_data = scaler.fit_transform(train_data)  # learns mean/std from train only
    test_scaled_data = scaler.transform(test_data)  # reuses train mean/std
    return train_scaled_data, test_scaled_data


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y)
scaler = StandardScaler()
X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/scaler.json", "w") as f:
    json.dump(
        {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
            "feature_names": list(X_train.columns),
        },
        f,
    )
