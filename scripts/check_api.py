import json

import requests

URL = "http://127.0.0.1:8000/predict"
with open("tests/samples.json") as f:
    samples = json.load(f)

for kind in ("signal", "background"):
    probs = []
    for row in samples[kind]:
        r = requests.post(URL, json={"features": row})
        r.raise_for_status()
        probs.append(r.json()["probability"])

    frac_signal = sum(p >= 0.5 for p in probs) / len(probs)
    print(
        f"{kind:10s} mean P(signal) = {sum(probs) / len(probs):.3f} | "
        f"classified as signal: {frac_signal:.0%}"
    )
