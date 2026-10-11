"""Save headless experiment figures from predictions and a model report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .utils import confusion_report


def generate_graphs(predictions, report, folder, model_name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    actual = predictions.label.to_numpy()
    predicted = predictions.prediction.to_numpy()
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [-1, 0, 1]).all():
        raise ValueError("Expected binary labels and predictions with optional -1 abstentions.")
    # Keep abstentions separate from negative predictions, including for attacks.
    labels = [0, 1, -1]
    names = ["Benign", "Attack", "Unscored"]
    matrix = confusion_matrix(actual, predicted, labels=labels)[:2]
    pd.DataFrame(matrix, index=names[:2], columns=names).to_csv(folder / "confusion_matrix.csv")
    (folder / "confusion_report.txt").write_text(confusion_report(
        actual, predicted, labels=labels, target_names=names, skip_y={"Unscored"}))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(2), names[:2])
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual label")
    ax.set_title(f"{model_name}: test confusion matrix")
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    loss = report["loss_history"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(range(1, len(loss) + 1), loss, marker="o", color="tab:green")
    ax.set(xlabel="Epoch", ylabel="Training loss",
           title=f"{model_name}: training loss")
    if len(loss) <= 15:
        ax.set_xticks(range(1, len(loss) + 1))
    ax.grid(alpha=0.25)
    fig.savefig(folder / "training_loss.png", dpi=160)
    plt.close(fig)

    timing = report["timing"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = ax.barh([key.replace("_seconds", "").replace("_", " ") for key in timing],
                   list(timing.values()), color="tab:blue")
    ax.bar_label(bars, fmt="%.2f s", padding=4)
    ax.set_xlim(0, max(max(timing.values()), 0.001) * 1.25)
    ax.set(xlabel="Seconds", title=f"{model_name}: recorded processing stages")
    fig.savefig(folder / "processing_time.png", dpi=160)
    plt.close(fig)
    print(f"Saved graphs and confusion report to {folder}", flush=True)
