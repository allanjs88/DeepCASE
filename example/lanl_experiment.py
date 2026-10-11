"""Shared inputs, metrics and output handling for LANL processing experiments.

This is not a matched-objective GNN ablation: the original interpreter receives
benign training scores, whereas the GNN uses validation-calibrated surprise.
The interval defaults deliberately surround a known attack for a smoke test;
they must not be presented as an independently selected evaluation interval.
"""
import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from deepcase_gnn.preprocessing import read_lanl


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def original_sequences(frame, vocabulary, length, window):
    """Original builder layout: left padding, newest event on the right.

    Use train-only tokens (PAD=0, UNK=1) and strictly earlier timestamps,
    matching GNN history boundaries. This replaces CLI preprocessing only.
    """
    tokens = frame.event.map(vocabulary).fillna(1).to_numpy(dtype=np.int64, copy=True)
    context = np.zeros((len(frame), length), dtype=np.int64)
    history = defaultdict(lambda: deque(maxlen=length))
    for timestamp, group in frame.groupby("timestamp", sort=True):
        for i, host in zip(group.index, group.source_computer):
            previous = history[host]
            while previous and previous[0][0] < timestamp - window:
                previous.popleft()
            if previous:
                context[i, -len(previous):] = [token for _, token in previous]
        # Insert only after scoring every simultaneous event.
        for i, host in zip(group.index, group.source_computer):
            history[host].append((timestamp, tokens[i]))
    return torch.from_numpy(context), torch.from_numpy(tokens).reshape(-1, 1)


def metrics(labels, prediction, scored):
    labels, prediction, scored = map(np.asarray, (labels, prediction, scored))
    positive, negative = labels == 1, labels == 0
    tp = int((positive & scored & (prediction == 1)).sum())
    fp = int((negative & scored & (prediction == 1)).sum())
    return {
        "test_events": len(labels), "positive_events": int(positive.sum()),
        "tp": tp, "fp": fp,
        "tn_scored": int((negative & scored & (prediction == 0)).sum()),
        "fn_scored": int((positive & scored & (prediction == 0)).sum()),
        "unscored_events": int((~scored).sum()),
        "unscored_positive_events": int((positive & ~scored).sum()),
        "unscored_negative_events": int((negative & ~scored).sum()),
        "attacks_not_alerted_including_unscored": int(positive.sum()) - tp,
        "coverage": float(scored.mean()),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_all_positives": tp / int(positive.sum()) if positive.any() else None,
        "false_alerts_per_negative_event": fp / int(negative.sum()) if negative.any() else None,
    }


def main(runners):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", type=Path, default=Path("example/data/lanl/auth_144000_154000.txt"))
    parser.add_argument("--redteam", type=Path, default=Path("example/data/lanl/redteam.txt"))
    parser.add_argument("--start", type=int, default=150820)
    parser.add_argument("--train-until", type=int, default=150850)
    parser.add_argument("--validation-until", type=int, default=150880)
    parser.add_argument("--end", type=int, default=150910)
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Read data and print split counts without training or writing files.")
    args = parser.parse_args()
    if not args.start < args.train_until < args.validation_until < args.end:
        parser.error("Require start < train-until < validation-until < end.")
    if min(args.epochs, args.window_seconds, args.max_length) < 1:
        parser.error("Epochs, window and length must be positive.")
    print("Reading bounded LANL source...", flush=True)
    started = time.perf_counter()
    frame = read_lanl(args.auth, args.redteam)
    frame = frame.loc[(frame.timestamp >= args.start) & (frame.timestamp < args.end)].reset_index(drop=True)
    train = frame.loc[frame.timestamp < args.train_until]
    validation = frame.loc[(frame.timestamp >= args.train_until) & (frame.timestamp < args.validation_until)]
    test = frame.loc[frame.timestamp >= args.validation_until]
    if any(part.empty for part in (train, validation, test)):
        parser.error("All three partitions must contain retained events.")
    if train.label.any():
        parser.error("This benign-training diagnostic requires no known attacks in training.")
    splits = {name: {"events": len(part), "positives": int(part.label.sum()),
                     "timestamps": int(part.timestamp.nunique())}
              for name, part in [("train", train), ("validation", validation), ("test", test)]}
    print(json.dumps(splits, indent=2), flush=True)
    if args.plan:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable.")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}. Choose a new directory.")
    args.output.mkdir(parents=True, exist_ok=False)
    status = {name: "pending" for name in runners}
    save_json(args.output / "status.json", status)
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "splits": splits, "read_seconds": time.perf_counter() - started,
                "torch": torch.__version__,
                "protocol": "Attack-centered processing pilot. Original benign-label interpreter vs GNN surprise; not an equal-supervision detection benchmark. Shared train-only vocabulary and strictly historical contexts; validation supplies history to original but calibrates GNN only."}
    save_json(args.output / "manifest.json", manifest)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part.to_csv(args.output / f"{name}.csv", index=False)
    vocabulary = {event: i + 2 for i, event in enumerate(sorted(train.event.unique()))}
    save_json(args.output / "vocabulary.json", vocabulary)
    reports = {}

    def timed(operation):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = operation()
        if args.device == "cuda":
            torch.cuda.synchronize()
        return result, time.perf_counter() - start

    for name in status:
        folder = args.output / name
        folder.mkdir()
        status[name] = "running"
        save_json(args.output / "status.json", status)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Starting {name}", flush=True)
        try:
            result, predicted, scored, loss, timings, extra = runners[name](
                args, frame, train, validation, test, vocabulary, folder, timed)
            assert result.event_id.tolist() == test.event_id.tolist(), "Prediction alignment mismatch"
            result.to_csv(folder / "predictions.csv", index=False)
            report = {"metrics": metrics(test.label.to_numpy(), predicted, scored),
                      "timing": timings, "loss_history": loss}
            report.update(extra)
            save_json(folder / "report.json", report)
            from reporting.experiment import generate_graphs
            generate_graphs(result, report, folder / "graphs", name)
            reports[name] = report
            status[name] = "complete"
        except BaseException as error:
            status[name] = f"failed: {type(error).__name__}: {error}"
            raise
        finally:
            save_json(args.output / "status.json", status)
    save_json(args.output / "comparison.json", reports)
    table = pd.DataFrame({name: {**r["metrics"], **r["timing"]} for name, r in reports.items()})
    table.to_csv(args.output / "comparison.csv")
    print(table.to_string(), flush=True)
    print(f"Saved comparison to {args.output}", flush=True)

