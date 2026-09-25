"""LANL adapter preserving both host endpoints and physical input-row IDs."""

import json

import numpy as np
import pandas as pd

AUTH_COLUMNS = ["timestamp", "source_user", "destination_user", "source_computer",
                "destination_computer", "authentication_type", "logon_type",
                "authentication_orientation", "success"]
REDTEAM_COLUMNS = ["timestamp", "source_user", "source_computer", "destination_computer"]
EVENT_FIELDS = ["authentication_type", "logon_type", "authentication_orientation",
                "success", "user_mismatch"]


def read_lanl(auth_path, redteam_path=None, nrows=None, auth_scope="remote-logon"):
    """Read a bounded LANL extract; zero labels mean no known red-team match.

    Event signatures remain strings until Preprocessor.fit; no full-data
    factorization. Raw account/host IDs are never embedded as categorical IDs.
    """
    auth = pd.read_csv(auth_path, names=AUTH_COLUMNS, keep_default_na=False,
                       nrows=nrows, dtype={c: "int64" if c == "timestamp" else "string"
                                          for c in AUTH_COLUMNS})
    auth["event_id"] = np.arange(len(auth)).astype(str)
    if auth_scope == "remote-logon":
        mask = (auth.source_computer.ne(auth.destination_computer)
                & ~auth.source_computer.isin(["", "?"])
                & ~auth.destination_computer.isin(["", "?"])
                & auth.authentication_orientation.eq("LogOn"))
        auth = auth.loc[mask].copy()
    elif auth_scope != "all":
        raise ValueError("auth_scope must be 'remote-logon' or 'all'.")
    # Even the all-events ablation requires identifiable endpoints.
    if auth[["source_computer", "destination_computer"]].isin(["", "?"]).any().any():
        raise ValueError("Unknown hosts cannot be connected as one shared '?' node.")
    auth["user_mismatch"] = auth.source_user.ne(auth.destination_user).astype(str)
    auth["event"] = [json.dumps(list(values), separators=(",", ":"))
                     for values in auth[EVENT_FIELDS].itertuples(index=False, name=None)]
    if redteam_path is not None:
        redteam = pd.read_csv(redteam_path, names=REDTEAM_COLUMNS, keep_default_na=False,
                             dtype={c: "int64" if c == "timestamp" else "string"
                                    for c in REDTEAM_COLUMNS})
        keys = set(redteam.itertuples(index=False, name=None))
        auth["label"] = [int(key in keys) for key in
                         auth[REDTEAM_COLUMNS].itertuples(index=False, name=None)]
    return auth.sort_values("timestamp", kind="stable").reset_index(drop=True)
