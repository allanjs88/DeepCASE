# Other imports
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

# DeepCASE Imports
from deepcase.preprocessing import Preprocessor
from deepcase import DeepCASE


AUTH_COLUMNS = [
    "timestamp",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "success",
]

REDTEAM_COLUMNS = [
    "timestamp",
    "source_user",
    "source_computer",
    "destination_computer",
]

DERIVED_COLUMNS = [
    "user_mismatch",
]

DEFAULT_EVENT_FIELDS = [
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "success",
    "user_mismatch",
]


def read_auth(path, nrows=None):
    """Read LANL authentication events."""
    auth = pd.read_csv(
        path,
        names=AUTH_COLUMNS,
        nrows=nrows,
        dtype={
            "timestamp": "int64",
            "source_user": "string",
            "destination_user": "string",
            "source_computer": "string",
            "destination_computer": "string",
            "authentication_type": "string",
            "logon_type": "string",
            "authentication_orientation": "string",
            "success": "string",
        },
    )
    auth["user_mismatch"] = (
        auth["source_user"] != auth["destination_user"]
    ).astype("string")
    return auth


def read_redteam(path):
    """Read LANL redteam events."""
    return pd.read_csv(path, names=REDTEAM_COLUMNS)


def redteam_keys(redteam):
    """Return redteam events as exact auth-row match keys."""
    return set(map(tuple, redteam[REDTEAM_COLUMNS].itertuples(index=False, name=None)))


def summarize_lanl(auth, redteam, top=5):
    """Print a compact analyst summary for the LANL sample."""
    print("\nLANL sample summary")
    print("  Auth time range: {} to {} seconds ({:.2f} to {:.2f} hours)".format(
        auth["timestamp"].min(),
        auth["timestamp"].max(),
        auth["timestamp"].min() / 3600,
        auth["timestamp"].max() / 3600,
    ))
    print("  Redteam events: {:,}".format(redteam.shape[0]))
    print("  Redteam time range: {} to {} seconds ({:.2f} to {:.2f} hours)".format(
        redteam["timestamp"].min(),
        redteam["timestamp"].max(),
        redteam["timestamp"].min() / 3600,
        redteam["timestamp"].max() / 3600,
    ))
    print("  Source/destination user mismatch: {:,}".format(
        int((auth["source_user"] != auth["destination_user"]).sum())
    ))

    for column in [
        "source_user",
        "destination_user",
        "source_computer",
        "destination_computer",
        "authentication_type",
        "logon_type",
        "authentication_orientation",
        "success",
    ]:
        print("\n  Top {} values for {}:".format(top, column))
        print(auth[column].value_counts(dropna=False).head(top).to_string())


def make_lanl_frame(auth_path, redteam_path, nrows=None, event_fields=None, summary=False):
    """Convert LANL auth data into DeepCASE's timestamp/machine/event/label CSV shape."""
    event_fields = event_fields or DEFAULT_EVENT_FIELDS
    redteam = read_redteam(redteam_path)
    redteam_events = redteam_keys(redteam)
    auth = read_auth(auth_path, nrows=nrows)

    auth = auth.sort_values("timestamp", kind="stable").reset_index(drop=True)
    auth["label"] = [
        int(key in redteam_events)
        for key in auth[REDTEAM_COLUMNS].itertuples(index=False, name=None)
    ]

    if summary:
        summarize_lanl(auth, redteam)

    # DeepCASE expects discrete event IDs. Factorize the selected auth fields
    # after joining them into one stable categorical value.
    event_values = auth[event_fields].astype("string").agg("|".join, axis=1)
    auth["event"] = pd.factorize(event_values, sort=True)[0]

    return auth.rename(
        columns={
            "source_computer": "machine",
        }
    )[["timestamp", "machine", "event", "label"]]


def partition_index(data, train_ratio=0.2, split_time=None):
    if not 0 < train_ratio < 1:
        raise ValueError("train-ratio must be between 0 and 1.")
    if data.empty:
        raise ValueError("No authentication rows loaded.")
    if split_time is None:
        candidate = min(int(len(data) * train_ratio), len(data) - 1)
        split_time = int(data.iloc[candidate]["timestamp"])
    split = int(data["timestamp"].searchsorted(split_time, side="left"))
    if split == 0 or split == len(data):
        raise ValueError("Temporal split produced an empty partition; choose another split-time.")
    print("Temporal split: train timestamp < {}; test timestamp >= {}".format(split_time, split_time))
    for name, frame in [("Train", data.iloc[:split]), ("Test", data.iloc[split:])]:
        positives = int(frame["label"].sum())
        print("{}: {:,} rows; {:,} redteam matches; timestamps {} to {}".format(
            name, len(frame), positives, frame["timestamp"].min(), frame["timestamp"].max()))
        if positives == 0:
            print("Warning: {} has no known positive events.".format(name))
    return split


def split_by_time(context, events, labels, train_ratio, split=None):
    if split is None:
        split = int(events.shape[0] * train_ratio)
    return (
        context[:split],
        context[split:],
        events[:split],
        events[split:],
        labels[:split],
        labels[split:],
    )


def parse_args():
    example_dir = Path(__file__).resolve().parent
    lanl_dir = example_dir / "data" / "lanl"

    parser = ArgumentParser(
        description="Read the LANL auth/redteam dataset and run a small DeepCASE example."
    )
    parser.add_argument(
        "--auth",
        default=lanl_dir / "auth.txt",
        type=Path,
        help="Path to auth.txt or auth.txt.gz.",
    )
    parser.add_argument(
        "--redteam",
        default=lanl_dir / "redteam.txt",
        type=Path,
        help="Path to redteam.txt or redteam.txt.gz.",
    )
    parser.add_argument(
        "--nrows",
        default=100_000,
        type=int,
        help="Number of auth rows to read. Use 0 to read the full file.",
    )
    parser.add_argument(
        "--event-fields",
        nargs="+",
        default=DEFAULT_EVENT_FIELDS,
        choices=AUTH_COLUMNS + DERIVED_COLUMNS,
        help="Auth columns used to define the discrete DeepCASE event type.",
    )
    parser.add_argument("--context-length", default=10, type=int)
    parser.add_argument("--timeout", default=86_400, type=int)
    parser.add_argument("--train-ratio", default=0.2, type=float)
    parser.add_argument("--split-time", type=int, help="First test timestamp; keeps equal timestamps together.")
    parser.add_argument("--report-only", action="store_true", help="Report partition label counts without allocating context tensors or training.")
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--query-batch-size", default=1024, type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nrows = None if args.nrows == 0 else args.nrows

    ########################################################################
    #                             Loading data                             #
    ########################################################################

    data = make_lanl_frame(
        auth_path=args.auth,
        redteam_path=args.redteam,
        nrows=nrows,
        event_fields=args.event_fields,
        summary=args.summary,
    )

    print("Loaded {:,} LANL auth events".format(data.shape[0]))
    print("Distinct DeepCASE events: {:,}".format(data["event"].nunique()))
    print("Redteam labels: {:,}".format(int(data["label"].sum())))
    if data["label"].sum() == 0:
        if nrows is None:
            print(
                "Warning: no exact redteam matches were found. Check that "
                "--auth and --redteam come from the same LANL release."
            )
        else:
            print(
                "Warning: no exact redteam matches were found in the first "
                "{:,} physical auth rows. LANL auth rows are sorted after "
                "loading, so a small --nrows sample may miss redteam events; "
                "increase --nrows or use --nrows 0 for the full file."
                .format(nrows)
            )

    split = partition_index(data, args.train_ratio, args.split_time)
    if args.report_only:
        raise SystemExit(0)

    preprocessor = Preprocessor(
        length=args.context_length,
        timeout=args.timeout,
    )

    context, events, labels, mapping = preprocessor.sequence(
        data=data,
        verbose=args.verbose,
    )

    if args.skip_model:
        print("Created context tensor {} and event tensor {}".format(
            tuple(context.shape),
            tuple(events.shape),
        ))
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    events = events.to(device)
    context = context.to(device)

    ########################################################################
    #                            Splitting data                            #
    ########################################################################

    (
        context_train,
        context_test,
        events_train,
        events_test,
        labels_train,
        labels_test,
    ) = split_by_time(context, events, labels, args.train_ratio, split=split)

    if events_train.shape[0] == 0 or events_test.shape[0] == 0:
        raise ValueError("Train/test split produced an empty partition.")

    ########################################################################
    #                            Using DeepCASE                            #
    ########################################################################

    deepcase = DeepCASE(
        features=len(mapping),
        max_length=args.context_length,
        hidden_size=args.hidden_size,
        eps=0.1,
        min_samples=5,
        threshold=0.2,
    )

    if torch.cuda.is_available():
        deepcase = deepcase.to("cuda")

    deepcase.fit(
        X=context_train,
        y=events_train.reshape(-1, 1),
        scores=labels_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        query_batch_size=args.query_batch_size,
        strategy="max",
        NO_SCORE=-1,
        verbose=True,
    )

    prediction = deepcase.predict(
        X=context_test,
        y=events_test.reshape(-1, 1),
        iterations=args.iterations,
        batch_size=args.query_batch_size,
        verbose=True,
    )

    ########################################################################
    #                          Perform evaluation                          #
    ########################################################################

    y_test = labels_test.cpu().numpy()
    y_pred = (prediction > 0).astype(int)

    print(classification_report(
        y_true=y_test,
        y_pred=y_pred,
        digits=4,
        zero_division=0,
    ))
