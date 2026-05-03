from pathlib import Path
import csv

import matplotlib.pyplot as plt


def plot_training_log(csv_path, output_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    epochs = [int(r["epoch"]) for r in rows]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for col in [c for c in rows[0].keys() if c.endswith("loss") or c == "loss"]:
        values = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
        if len(values) == len(epochs):
            plt.plot(epochs, values, label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)
