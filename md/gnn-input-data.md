# Input data for DeepCASE with a GNN

## Purpose and status

This document proposes an input design for extending DeepCASE with graph neural
networks while retaining support for LANL, HDFS, and other datasets. It describes
design options, not an implemented graph pipeline or completed experiments.

The central proposal is to keep a common event table and introduce a separate
graph-building stage. Dataset adapters normalize records; graph builders decide
which relationships are available and how to represent them.

Related code and planning:

- [HDFS example](../example/example_hdfs.py)
- [LANL example](../example/example_lanl.py)
- [Current preprocessor](../deepcase/preprocessing/preprocessor.py)
- [LANL implementation and experiment plan](implementation-experiments.md)

## 1. Choose what the graph represents

Two graph designs answer different research questions:

| Design | Nodes | Edges | Dataset support |
| --- | --- | --- | --- |
| Event graph | Individual event occurrences | Temporal succession; optional shared-entity relationships | HDFS and LANL |
| Host graph | Computers | Historical source-to-destination authentications | LANL and other datasets with endpoints |

The existing experiment plan proposes **GRU + GNN over hosts**. This augments
temporal representations with cross-host information. The current HDFS sequence
files cannot supply those host relationships.

A temporal event graph is a portable alternative for testing a GNN sequence
encoder on the same histories as DeepCASE. These options should remain separate
experiments: replacing the encoder and adding host topology change different
parts of the system.

## 2. Common event-table schema

Normalize each dataset into one row per event occurrence:

| Field | Purpose | HDFS | LANL |
| --- | --- | --- | --- |
| `event_id` | Stable occurrence identifier | Generated from line and position | Original input row identifier |
| `timestamp` | Ordering or time | Synthetic position | Authentication timestamp |
| `sequence_id` | Group for temporal history | Input line number | Source computer |
| `event_type` | Categorical event type | Existing integer | Encoded authentication attributes |
| `source` | Optional source entity | Unavailable | Source computer |
| `destination` | Optional destination entity | Unavailable | Destination computer |
| `label` | Optional score or class | Unavailable in the current input file | Exact redteam match |

`event_id` identifies an occurrence; `event_type` identifies a category. Two
occurrences can share an event type but must remain distinct graph nodes.

`sequence_id` generalizes the current `machine` column: an HDFS line is an
independent sequence, not necessarily a physical computer. Store metadata stating
whether timestamps are seconds or synthetic positions. Do not interpret HDFS
position differences as physical elapsed time.

Preserve additional LANL fields, including users and authentication attributes,
in the derived dataset even if a particular model does not consume them. Missing
endpoint information should remain missing; do not invent host relationships.

For compatibility with the existing preprocessor, create a separate view:

```python
baseline_data = records.rename(columns={
    "sequence_id": "machine",
    "event_type": "event",
})[["timestamp", "machine", "event"]].copy()

if "label" in records.columns:
    baseline_data["label"] = records["label"].to_numpy()

baseline_data = baseline_data.reset_index(drop=True)
context, events, labels, mapping = preprocessor.sequence(baseline_data)
```

Keep the stable occurrence IDs separately aligned with the output rows. The
current preprocessor mutates the event column during remapping and uses DataFrame
indices to place contexts, hence the copy and contiguous index above. This example
illustrates schema compatibility, not a complete training-vocabulary solution.

## 3. LANL transformation example

### Raw authentication records

The current LANL reader expects headerless CSV with these fields:

```text
timestamp,source_user,destination_user,source_computer,destination_computer,authentication_type,logon_type,authentication_orientation,success
```

Illustrative records:

```text
100,U1@D,U1@D,C1,C2,Kerberos,Network,LogOn,Success
105,U2@D,U3@D,C2,C3,NTLM,Network,LogOn,Fail
110,U1@D,U1@D,C1,C3,Kerberos,Network,LogOn,Success
120,U1@D,U4@D,C1,C4,NTLM,Network,LogOn,Fail
```

### Event categories

The existing adapter derives `user_mismatch` and combines these default fields:

```text
authentication_type | logon_type | authentication_orientation | success | user_mismatch
```

For this example, the categories are:

```text
Kerberos|Network|LogOn|Success|False -> 0
NTLM|Network|LogOn|Fail|True         -> 1
```

These numbers are illustrative categorical IDs, not numeric measurements. For
formal experiments, fit the category vocabulary on training data and reserve an
unknown category for unseen validation/test combinations. The current example's
factorization over all loaded rows needs adaptation for that protocol.

### Labels and normalized records

If a redteam record contains:

```text
120,U1@D,C1,C4
```

the exact tuple `(timestamp, source_user, source_computer, destination_computer)`
matches the final authentication. The normalized table is:

| event_id | timestamp | sequence_id | event_type | source | destination | label |
| --- | ---: | --- | ---: | --- | --- | ---: |
| e0 | 100 | C1 | 0 | C1 | C2 | 0 |
| e1 | 105 | C2 | 1 | C2 | C3 | 0 |
| e2 | 110 | C1 | 0 | C1 | C3 | 0 |
| e3 | 120 | C1 | 1 | C1 | C4 | 1 |

Zero means no known exact redteam match, not independently verified benign.
Labels are supervision/evaluation data and must not determine graph structure or
enter node/edge features.

The current LANL converter returns only `timestamp`, `machine`, `event`, and
`label`. Graph construction requires preserving source and destination endpoints
before that reduction.

## 4. Portable temporal event graphs

For predicting `e3`, the source-local history from the LANL example is:

```text
e0 -> e2       target: event_type(e3) = 1
```

A minimal graph representation is:

```python
node_event_type = [0, 0]  # e0 and e2
edge_index = [
    [0],                 # Source node indices
    [1],                 # Destination node indices
]
target_event_type = 1
```

Event types become learned embeddings before message passing. A readout converts
the node representations into a graph/context representation for next-event
prediction. Define the readout and message-passing depth explicitly: on a chain,
a shallow GNN with only last-node readout cannot access arbitrarily distant events.

A proposed sample interface is:

| Component | Shape | Meaning |
| --- | --- | --- |
| `node_event_type` | `(V,)` | Integer category for each historical occurrence |
| `edge_index` | `(2, E)` | Directed edges using local node indices |
| `node_position` | `(V,)` | Position within the historical context |
| `edge_type` | `(E,)`, optional | Temporal or other relation category |
| `edge_attr` | `(E, F_e)`, optional | Available historical edge features |
| `target_event_type` | Scalar | Next-event training target |
| `label` | Scalar, optional | Separate security score/class |
| `target_event_id` | Metadata | Stable alignment with original records |

`V` is the number of history nodes, `E` the number of edges, and `F_e` the edge
feature dimension. Batched graphs additionally require graph-membership metadata
and node-index offsets. This is a proposed interface, not an existing repository
API.

For the first controlled comparison, use exactly the same preceding events,
history length, timeout, and ordering policy for both models:

```text
DeepCASE:   event history -> GRU -> next-event prediction
GNN:        same history -> temporal graph -> GNN -> next-event prediction
```

Directed edges and position features preserve ordering information. Do not turn
padding tokens into ordinary event nodes. Define an explicit empty-history
representation for the first event in a sequence.

### Optional LANL relationships

A later variant can expand the historical neighborhood and add typed relations:

```text
e0 -> e1    destination(e0) equals source(e1): shared host C2
e0 -> e2    consecutive events from source C1
```

This requires including `e1`, which is absent from the source-local history for
`e3`. It therefore changes both the graph connectivity and available information.
Report it as a separate experiment. Shared entities indicate observed
relationships, not established causation or a confirmed attack path.

Bound the time window and neighborhood size so repeated entities do not create
an unmanageably dense graph.

## 5. HDFS transformation example

The current text input already contains integer event sequences:

```text
5 22 11 9
5 11 9
```

The adapter can produce:

| event_id | timestamp | sequence_id | event_type |
| --- | ---: | ---: | ---: |
| h0 | 0 | 0 | 5 |
| h1 | 1 | 0 | 22 |
| h2 | 2 | 0 | 11 |
| h3 | 3 | 0 | 9 |
| h4 | 4 | 1 | 5 |
| h5 | 5 | 1 | 11 |
| h6 | 6 | 1 | 9 |

To predict `h3`:

```text
History:          h0 -> h1 -> h2
Event categories: 5     22    11
Target:           9
```

To predict `h6`:

```text
History:          h4 -> h5
Event categories: 5     11
Target:           9
```

Each line has its own history. The graph builder must not connect the end of one
line to the beginning of the next. Original event categories are encoded into a
training vocabulary before embedding lookup.

This supports the same temporal event-graph architecture as LANL, trained with
the dataset's own vocabulary. It does not imply sharing learned weights or
inventing cross-host relationships. The current HDFS file supplies neither real
timestamps nor labels nor source/destination endpoints.

## 6. LANL host graphs: the planned hybrid

The existing experiment plan instead proposes a graph of computers. For scoring
`e3` at time 120, earlier observations yield:

```text
C1 -> C2
C2 -> C3
C1 -> C3
```

| Component | Proposed definition |
| --- | --- |
| Nodes | Source and destination computers |
| Node features | GRU representation of each host's preceding local events |
| Edges | Historical source-to-destination authentications |
| Repeated edges | Aggregate within the history window |
| Edge features | Counts, success statistics, recency; optional type information |
| Scored event | Current authentication attributes and its source/destination |
| Prediction | Event-level detection score |

For the current event `C1 -> C4`, initialize C4 explicitly if it has no history.
Do not insert the current authentication into historical message passing under
the event-arrival protocol. Its observed attributes can be supplied separately to
the detection head.

This GNN augments the GRU. It cannot be evaluated as the same host-graph model on
the existing HDFS sequence files without additional endpoint data.

## 7. Comparison and leakage controls

| Experiment | Information | Datasets | Research question |
| --- | --- | --- | --- |
| Original DeepCASE | Local event sequences | HDFS, LANL | Baseline performance |
| Temporal event GNN | Same local histories | HDFS, LANL | Effect of changing the sequence encoder |
| Event GNN with entity relations | Expanded histories and endpoints | LANL or equivalent | Value of relational information |
| GRU + host GNN | Temporal representations and host topology | LANL or equivalent | Value of the planned hybrid |

For the hybrid, also compare a GRU with the same scoring head but no message
passing, as proposed in the existing experiment plan. Otherwise, improvements may
come from changing the scorer rather than adding graph information.

Keep these controls explicit:

- Use the same chronological partitions, scored records, label budget, and tuning
  effort for paired models.
- Fit vocabularies and preprocessing parameters on training data only.
- For LANL event-arrival scoring, build history from strictly earlier timestamps;
  do not treat file order within a second as known physical order. Apply the same
  tie policy to the sequence baseline. For HDFS, use the supplied sequence order.
- Never include future observations or target labels in graph construction.
- For next-event prediction, exclude the target event type from input features.
  For detection at event arrival, current authentication attributes may be observed
  inputs, but the security label remains hidden.
- Document history length, time window, relation rules, edge direction, readout,
  empty histories, unseen categories, and unseen-host initialization.
- Preserve stable event identifiers so predictions and labels remain aligned.

The current HDFS example reports next-event prediction; the LANL example reports
redteam detection. These are different tasks. Compare architectures within each
dataset/task and do not interpret their metrics as interchangeable.

## 8. DeepCASE integration boundary

A GNN encoder is not automatically a drop-in replacement for all of DeepCASE.
The interpreter relies on context-building and attention/query behavior. Either
preserve those interfaces or describe and evaluate the changed interpreter as an
additional methodological change.

A practical staged approach is:

1. Implement dataset adapters that preserve the common event records and metadata.
2. Establish the original sequence baseline with a fixed evaluation protocol.
3. Build temporal event graphs from identical histories for HDFS and LANL.
4. Compare next-event prediction before claiming full interpreter compatibility.
5. Integrate and evaluate the interpreter or a clearly specified detection head.
6. Add LANL entity relationships or the planned GRU + host GNN as separate variants.

This keeps dataset compatibility independent of graph semantics and separates
the benefit of a new encoder from the benefit of additional relational data.
