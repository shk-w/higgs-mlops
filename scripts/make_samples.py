import json

from higgs.data import X_test, y_test  # runs the data pipeline

# RAW features, not X_test_scaled: the API does the scaling itself
signal = X_test[y_test == 1].head(50)
background = X_test[y_test == 0].head(50)

with open("tests/samples.json", "w") as f:
    json.dump({"signal": signal.values.tolist(), "background": background.values.tolist()}, f)
print("saved", len(signal), "signal and", len(background), "background rows")
