"""Run the requested HDFS/LANL epoch experiments on CUDA and save plots/logs."""
import json
import os
from pathlib import Path
import subprocess
import sys

EXAMPLE = Path(__file__).resolve().parent
ROOT = EXAMPLE.parent
RESULTS = EXAMPLE / "results"


def experiment(dataset, epochs):
    import random
    import runpy
    import numpy as np
    import torch
    from sklearn.metrics import classification_report
    from plotting import ResultsPlotter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; refusing to fall back to CPU.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = RESULTS / dataset / str(epochs)
    output.mkdir(parents=True, exist_ok=True)
    script = EXAMPLE / f"example_{dataset}.py"
    args = [str(script), "--epochs", str(epochs)]
    if dataset == "hdfs":
        args += ["--nrows", "1000", "--no-graphs"]
    else:
        args += ["--auth", "example/data/lanl/auth_150000_154000.txt",
                 "--redteam", "example/data/lanl/redteam.txt", "--nrows", "0",
                 "--split-time", "151800", "--batch-size", "32",
                 "--query-batch-size", "128"]
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Arguments:", args, flush=True)
    sys.argv = args
    state = runpy.run_path(str(script), run_name="__main__")
    builder = (state["context_builder"] if dataset == "hdfs"
               else state["deepcase"].context_builder)
    actual, predicted = state["y_test"], state["y_pred"]
    np.savez_compressed(output / "predictions.npz", y_test=actual, y_pred=predicted)
    metrics = classification_report(actual, predicted, output_dict=True, zero_division=0)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    losses = [float(value) for value in builder.loss_history]
    (output / "loss_history.json").write_text(json.dumps(losses, indent=2))
    ResultsPlotter.plot_prediction_results(
        actual, predicted, output / f"{dataset}_prediction_results.png", show=False,
        title=f"{dataset.upper()} — {epochs} epochs")
    ResultsPlotter.plot_training_loss_history(
        losses, output / f"{dataset}_training_loss_summary.png", show=False)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepcase-matplotlib")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if len(sys.argv) == 3:
        experiment(sys.argv[1], int(sys.argv[2]))
    else:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU unavailable. No experiments started.")
        for dataset in ("hdfs", "lanl"):
            for epochs in (25, 50, 100):
                output = RESULTS / dataset / str(epochs)
                output.mkdir(parents=True, exist_ok=True)
                print(f"Running {dataset}, {epochs} epochs: {output}", flush=True)
                with (output / "run.log").open("w") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    dataset, str(epochs)], stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
