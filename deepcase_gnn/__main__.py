"""Run a bounded LANL pilot or a small synthetic end-to-end demonstration."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch_geometric

from . import DeepCASEGNN
from .interpreter import detection_metrics
from .preprocessing import read_lanl


def demo_frame():
    rows = []
    for i in range(90):
        attack = i >= 70 and i % 7 == 0
        rows.append(dict(event_id=str(i), timestamp=i, source_computer=f"C{i % 3}",
                         destination_computer=f"C{(i + 1) % 3}", source_user=f"U{i % 2}",
                         event="Fail" if attack else "Success", success="Fail" if attack else "Success",
                         label=int(attack)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--auth", type=Path, help="Headerless LANL auth file (plain or gzip).")
    parser.add_argument("--redteam", type=Path)
    parser.add_argument("--nrows", type=int, help="Physical row prefix limit; pilot only.")
    parser.add_argument("--auth-scope", choices=["remote-logon", "all"], default="remote-logon")
    parser.add_argument("--train-until", type=float, help="Training timestamps < this cutoff.")
    parser.add_argument("--validation-until", type=float, help="Test timestamps >= this cutoff.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8, help="Timestamp snapshots per batch.")
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=2, help="Use 0 for the matched GRU-only ablation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.demo:
        frame = demo_frame()
        train_until = args.train_until if args.train_until is not None else 50
        validation_until = args.validation_until if args.validation_until is not None else 70
    else:
        if args.train_until is None or args.validation_until is None:
            parser.error("LANL requires explicit --train-until and --validation-until timestamps.")
        frame = read_lanl(args.auth, args.redteam, args.nrows, args.auth_scope)
        train_until, validation_until = args.train_until, args.validation_until
    if not train_until < validation_until:
        parser.error("--train-until must precede --validation-until.")
    train = frame.loc[frame.timestamp < train_until]
    validation = frame.loc[(frame.timestamp >= train_until) & (frame.timestamp < validation_until)]
    test = frame.loc[frame.timestamp >= validation_until]
    if any(part.empty for part in (train, validation, test)):
        parser.error("All three chronological partitions must contain events; adjust the pilot interval/cutoffs.")
    model = DeepCASEGNN(max_length=args.max_length, window_seconds=args.window_seconds,
                        hidden_size=args.hidden_size, gnn_layers=args.gnn_layers,
                        seed=args.seed, device=args.device)
    model.fit(train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    model.calibrate(validation, history=train, quantile=args.quantile, batch_size=args.batch_size)
    result = model.predict(test, history=pd.concat([train, validation]), batch_size=args.batch_size)
    report = {"synthetic_demo": args.demo, "config": model.config,
              "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                           "learning_rate": 0.001, "quantile": args.quantile},
              "source": {"auth": str(args.auth) if args.auth else None,
                         "redteam": str(args.redteam) if args.redteam else None,
                         "nrows": args.nrows, "auth_scope": args.auth_scope},
              "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__},
              "train_until": train_until, "validation_until": validation_until,
              "partitions": {name: {"events": len(part),
                                     "positives": int(part.label.sum()) if "label" in part else None}
                             for name, part in [("train", train), ("validation", validation), ("test", test)]},
              "threshold": model.interpreter.threshold, "loss_history": model.loss_history}
    if "label" in result:
        report["test_metrics"] = detection_metrics(result.label, result.score, result.prediction)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.pt")
    result.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved model, predictions and report to {args.output}")


if __name__ == "__main__":
    main()
