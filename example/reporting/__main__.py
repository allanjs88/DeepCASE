"""Regenerate graphs from saved results without retraining."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .experiment import generate_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Model folder containing predictions.csv and report.json")
    args = parser.parse_args()
    generate_graphs(pd.read_csv(args.folder / "predictions.csv"),
                    json.loads((args.folder / "report.json").read_text()),
                    args.folder / "graphs", args.folder.name)


if __name__ == "__main__":
    main()
