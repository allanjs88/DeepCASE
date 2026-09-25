# DeepCASE-GNN v0.1: GRU + GraphSAGE

An independent `deepcase_gnn` Python package beside `deepcase`. The underscore is
required for ordinary Python imports. The original model is unchanged. This
package follows the thesis proposal's sections 4.1.1 and A.3: host-local GRU
representations feed a directed host authentication graph and an event scorer.
It is a working pilot, not the completed thesis evaluation or a drop-in
replacement for DeepCASE's attention/query interpreter.

## Architecture and explicit first-version decisions

```text
LANL authentication rows
  -> chronological train / validation / test split
  -> training-only event vocabulary
  -> each timestamp t: historical host graph over [t - window_seconds, t)
       node = computer
       host's last max_length outgoing event categories -> embedding -> GRU
       GRU output + host statistics -> projection -> GraphSAGE (2 layers)
  -> source embedding + destination embedding + historical pair statistics
  -> event-category logits
  -> surprise = -log P(observed event category | history, endpoints)
  -> validation-quantile threshold -> event-level anomaly prediction
```

- Hosts have no trainable ID embeddings. New hosts receive empty histories and
  zero statistics. Incoming neighbors can supply information if present.
- Host statistics: log1p counts of outgoing events, distinct source users,
  successes, and failures within the time window. The GRU sequence is capped at
  `max_length`, while statistics and edges use the entire time window.
- Historical edges are directed source -> destination, deduplicated by pair.
  PyG `SAGEConv(aggr="mean")` aggregates incoming neighbors with its root term;
  reverse edges are not automatically added.
- Graphs retain authentication counts as `edge_weight` and four edge attributes:
  log1p count, log1p distinct users, failure fraction, and normalized age of the
  most recent authentication. **Standard SAGEConv ignores these attributes and
  weights.** The scored pair's attributes enter the event head. Weighted or
  edge-conditioned message passing is a future extension, not claimed here.
- The current event's category is a prediction target, never a model input. The
  observed source/destination endpoints select embeddings; this is conditional
  event-arrival scoring, not a probability model over all possible host pairs.
  There is no separate novelty score for an unseen destination or user identity.
- Training uses categorical cross-entropy, jointly updating GRU, GraphSAGE and
  scorer. Security labels are evaluation metadata only. Training data is assumed
  to represent background behavior; it is not filtered using red-team labels.
- This reimplements a GRU encoder; it does not import pretrained DeepCASE weights,
  preserve the original attention decoder, cluster fingerprints, or propagate
  analyst labels. The `Interpreter` name here means a validation-calibrated
  surprise threshold. A high score is not a calibrated attack probability.
- `gnn_layers=0` gives a GRU-only ablation with identical preprocessing and scoring
  head. Run the original DeepCASE separately as the thesis baseline; this ablation
  isolates message passing but is not the original DeepCASE baseline.

## Install and run

From the `DeepCASE` repository, using the existing environment:

```bash
.venv/bin/python -m pip install -r requirements-gnn.txt
.venv/bin/python -m deepcase_gnn --demo --epochs 3 --output /tmp/deepcase-gnn-demo
```

The demo needs no dataset download and verifies the complete training, calibration,
prediction and checkpoint workflow. Its artificial labels and simple distribution
do not establish detection performance. Use `OMP_NUM_THREADS=1` for small CPU
pilots if thread overhead is high.

For a new environment, first install the baseline numerical dependencies and the
appropriate CPU/CUDA build of PyTorch, then this optional requirement. The repository
also supports `pip install -e '.[gnn]'` when the baseline dependencies are present.
No compiled PyG sampling extensions are needed for this version.

Example LANL pilot (choose cutoffs spanned by your contiguous input interval):

```bash
.venv/bin/python -m deepcase_gnn \
  --auth /path/to/auth.txt.gz --redteam /path/to/redteam.txt.gz \
  --train-until 120000 --validation-until 147600 \
  --window-seconds 300 --max-length 10 --epochs 10 \
  --batch-size 4 --device cpu --seed 0 --output /tmp/deepcase-gnn-lanl
```

Cutoffs are absolute LANL timestamp seconds: train < 120000, validation in
[120000,147600), test >= 147600 in this example. These are example experiment
settings, not tuned or validated choices. All partitions must be nonempty.
`--nrows` selects a physical row prefix for debugging only; it does not guarantee
temporal coverage. The default scope matches the current LANL baseline adapter:
remote `LogOn` records with distinct known hosts. `--auth-scope all` includes
local/other authentication events but rejects unknown host endpoints.

Outputs: `model.pt`, `predictions.csv` with stable event IDs, and `report.json`
containing configuration, versions, split/label counts, loss, threshold and metrics.
Metrics are precision, recall, trapezoidal PR AUC, average precision (separately
named), and false positive rate. Recall and PR metrics are null without positives;
FPR is null without negatives. A zero label means no known red-team match.

For a matched GRU-only run, repeat the command with `--gnn-layers 0` and a different
output directory. Keep data, split, seed and training/calibration settings equal.

## Python API

```python
import pandas as pd
from deepcase_gnn import DeepCASEGNN
from deepcase_gnn.preprocessing import read_lanl

frame = read_lanl("/path/to/auth.txt.gz", "/path/to/redteam.txt.gz")
train = frame[frame.timestamp < 120000]
validation = frame[(frame.timestamp >= 120000) & (frame.timestamp < 147600)]
test = frame[frame.timestamp >= 147600]

model = DeepCASEGNN(max_length=10, window_seconds=300, gnn_layers=2)
model.fit(train, epochs=10, batch_size=4)
model.calibrate(validation, history=train, quantile=0.99)
predictions = model.predict(test, history=pd.concat([train, validation]))
model.save("model.pt")
restored = DeepCASEGNN.load("model.pt", device="cpu")
```

Generic input is a DataFrame with `timestamp` (numeric seconds), `event` (category),
`source_computer`, and `destination_computer`. Supply unique `event_id` values before
splitting; otherwise original DataFrame index values are used as IDs. Optional
`source_user` and `success` (`Success`/`Fail`) enable statistics, and `label` (0/1)
is retained for evaluation. Outputs are chronological; join by `event_id` to restore
the original order. This host-graph model requires host endpoints; HDFS sequence
files alone do not supply them.

## Temporal and evaluation controls

- Vocabulary is fitted on training only. Padding and unseen categories use
  separate reserved tokens. Unseen categories are scored as UNK, whose predictive
  calibration is limited because training has no unknown targets by default.
- No current, simultaneous, or future event enters a target's graph, sequence,
  or aggregate features. Earlier ties use stable input order for reproducible
  sequence truncation; that ordering does not establish physical causality.
- One graph snapshot is built per distinct target timestamp; tied target events
  share it. Their endpoints may add isolated nodes but no current edges or features.
  Every target row, including duplicates, contributes once to the loss and metrics.
- Earlier training/validation observations can be passed as explicit history.
  Their latest timestamp must precede all scored rows. Earlier test observations
  also become history for later test events, without labels or weight updates.
- Calibration uses only validation scores; test labels do not select thresholds.
  The quantile controls a validation exceedance budget, not a guaranteed test FPR.
- Red-team keys match exactly `(timestamp, source_user, source_computer,
  destination_computer)` and do not define the graph. Account identity is not
  currently modeled beyond user counts and the event's source/destination-user
  mismatch category.

## Limits and validation

This first version loads a bounded event table in memory and constructs each
timestamp's historical graph lazily. It uses full-neighborhood GraphSAGE, **not
neighbor sampling**, and repeats overlapping window construction. Large windows,
many timestamps, or many events per second can be expensive. `batch_size` counts
snapshots, not nodes or target events. Start with a contiguous small interval;
full LANL training needs streaming/caching and sampling work. GPU execution is
available through `--device cuda` when supported by the environment.

Run the regression tests from the repository:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_deepcase_gnn.py' -v
```

Tests cover temporal leakage, tied timestamps, window expiry, duplicate edges,
unknown hosts/categories, train-only vocabulary, PyG endpoint offsets, batching
equivalence, gradients through both encoders, topology sensitivity, label-free
training, checkpoint round trips, empty histories, ablation and metrics.

Follow-up thesis work includes original-interpreter integration or a justified
alternative, weighted message passing, sampling, full provenance/coverage reports,
and repeated paired experiments across at least ten seeds. No performance
improvement over DeepCASE is claimed by this implementation.

References: [PyG introduction](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html),
[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html),
[thesis proposal](../../DeepCASE-GNN-latex/thesis-proposal/thesis-proposal.pdf).
