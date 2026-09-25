"""Training-only vocabulary and strictly historical sliding host graphs."""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


REQUIRED = {"timestamp", "event", "source_computer", "destination_computer"}


def validate_frame(frame):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError("Event table must not be empty.")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Required event fields must not contain nulls.")
    frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="raise")
    if not np.isfinite(frame.timestamp.to_numpy()).all():
        raise ValueError("Timestamps must be finite numeric seconds.")
    for column in ("source_computer", "destination_computer", "event"):
        frame[column] = frame[column].astype(str)
    for column in ("source_computer", "destination_computer"):
        if frame[column].isin(["", "?"]).any():
            raise ValueError("Host endpoints must be known, nonempty identifiers.")
    if "event_id" not in frame:
        # Prefixes/partitions should be assigned IDs before splitting the input.
        frame["event_id"] = frame.index.astype(str)
    frame["event_id"] = frame.event_id.astype(str)
    if frame.event_id.duplicated().any() or frame.event_id.isna().any():
        raise ValueError("event_id must identify each physical row uniquely.")
    if "label" in frame and not frame.label.isin([0, 1]).all():
        raise ValueError("Optional security labels must be 0 or 1.")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


class Preprocessor:
    """Build host graphs in [t - window_seconds, t), excluding all ties at t.

    Node sequences are the last `length` outgoing events of each host in that
    window. Padding=0 and unknown=1; no trainable host identity lookup is used.
    Fit once on training rows, then transform later partitions with explicit
    earlier history. Security labels are metadata only.
    """

    def __init__(self, length=10, window_seconds=300):
        if not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer.")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive.")
        self.length = length
        self.window_seconds = float(window_seconds)
        self.vocabulary = None

    @property
    def features(self):
        if self.vocabulary is None:
            raise RuntimeError("Fit the vocabulary on training data first.")
        return len(self.vocabulary) + 2

    def fit(self, frame):
        frame = validate_frame(frame)
        self.vocabulary = {value: i + 2 for i, value in enumerate(sorted(frame.event.unique()))}
        return self

    def transform(self, frame, history=None):
        self.features  # Check fitted state even for an empty vocabulary.
        frame = validate_frame(frame)
        if history is not None and not history.empty:
            history = validate_frame(history)
            if history.timestamp.max() >= frame.timestamp.min():
                raise ValueError("Explicit history must be strictly earlier than scored rows.")
            if set(history.event_id) & set(frame.event_id):
                raise ValueError("History and scored rows must have distinct event IDs.")
            combined = pd.concat([history, frame], ignore_index=True)
            offset = len(history)
        else:
            combined, offset = frame, 0
        combined = combined.copy()
        combined["token"] = combined.event.map(self.vocabulary).fillna(1).astype(int)
        return HostGraphDataset(combined, offset, self.length, self.window_seconds)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


class HostGraphDataset(Dataset):
    """Lazily materialize one graph per distinct scoring timestamp.

    `target_index` is [2, events] so PyG offsets both endpoint indices when
    batching snapshots. `row_position` maps outputs to `frame` (scored rows).
    Each event is scored exactly once, including duplicate authentication rows.
    """

    def __init__(self, combined, offset, length, window_seconds):
        self.combined = combined.reset_index(drop=True)
        self.frame = self.combined.iloc[offset:].reset_index(drop=True)
        self.offset = offset
        self.length = length
        self.window_seconds = window_seconds
        self.times = self.combined.timestamp.to_numpy()
        self.groups = []
        for timestamp, group in self.frame.groupby("timestamp", sort=True):
            positions = group.index.to_numpy()
            self.groups.append((timestamp, positions))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        timestamp, positions = self.groups[index]
        left = self.times.searchsorted(timestamp - self.window_seconds, side="left")
        right = self.times.searchsorted(timestamp, side="left")
        historical = self.combined.iloc[left:right]
        targets = self.frame.iloc[positions]
        hosts = sorted(set(historical.source_computer) | set(historical.destination_computer)
                       | set(targets.source_computer) | set(targets.destination_computer))
        host_id = {host: i for i, host in enumerate(hosts)}
        sequence = torch.zeros((len(hosts), self.length), dtype=torch.long)
        lengths = torch.zeros(len(hosts), dtype=torch.long)
        stats = torch.zeros((len(hosts), 4), dtype=torch.float32)
        for host, rows in historical.groupby("source_computer", sort=False):
            node = host_id[host]
            tokens = rows.token.to_numpy()[-self.length:]
            lengths[node] = len(tokens)
            sequence[node, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            users = rows.source_user.nunique() if "source_user" in rows else 0
            success = int(rows.success.eq("Success").sum()) if "success" in rows else 0
            failed = int(rows.success.eq("Fail").sum()) if "success" in rows else 0
            stats[node] = torch.log1p(torch.tensor([len(rows), users, success, failed], dtype=torch.float))

        edges, attributes, weights = [], [], []
        pair_attributes = defaultdict(lambda: [0.0] * 4)
        for (source, destination), rows in historical.groupby(
                ["source_computer", "destination_computer"], sort=True):
            users = rows.source_user.nunique() if "source_user" in rows else 0
            failed = float(rows.success.eq("Fail").mean()) if "success" in rows else 0.0
            attr = [math.log1p(len(rows)), math.log1p(users), failed,
                    float(timestamp - rows.timestamp.max()) / self.window_seconds]
            edges.append([host_id[source], host_id[destination]])
            attributes.append(attr)
            weights.append(len(rows))
            pair_attributes[(source, destination)] = attr

        endpoints = list(zip(targets.source_computer, targets.destination_computer))
        graph = Data(
            num_nodes=len(hosts),
            sequence=sequence,
            lengths=lengths,
            x=stats,
            edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            edge_attr=torch.tensor(attributes, dtype=torch.float32).reshape(-1, 4),
            edge_weight=torch.tensor(weights, dtype=torch.float32),
            target_index=torch.tensor([[host_id[s], host_id[d]] for s, d in endpoints],
                                      dtype=torch.long).t().contiguous(),
            target_attr=torch.tensor([pair_attributes[pair] for pair in endpoints], dtype=torch.float32),
            y=torch.tensor(targets.token.to_numpy(), dtype=torch.long),
            row_position=torch.tensor(positions, dtype=torch.long),
        )
        return graph
