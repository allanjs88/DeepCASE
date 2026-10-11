"""Run both independent LANL experiments with shared inputs and comparison outputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lanl_experiment import main as run_experiments, metrics, original_sequences
from experiment_deepcase_lanl import run as run_original
from experiment_deepcase_gnn_lanl import run as run_gnn


def main():
    run_experiments({"original": run_original, "gnn": run_gnn})


if __name__ == "__main__":
    main()
