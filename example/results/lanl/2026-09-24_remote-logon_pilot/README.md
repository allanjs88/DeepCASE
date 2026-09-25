# Remote LogOn pilot — 2026-09-24

New experiments are isolated here; the earlier `../25`, `../50`, and `../100`
results are historical and remain untouched.

Protocol: extract every authentication row in [144000, 154000) from the complete
local source, then retain LogOn between distinct known hosts. Retain all account
categories, authentication packages, outcomes, and duplicate events. Match exact
original Red Team keys. Training is [144000, 147600), test is [147600, 154000).
This is a bounded development pilot with only one hour of training history, not
training on the full first 41 hours or evaluating the full remainder of LANL.
The test endpoint reuses the earlier development interval; results are not final
evaluation evidence. No training attack labels are allowed.

Completed independent epoch runs: 25, 50, 100. See [summary.md](summary.md) for verified results. Seed 42, batch size 128, query batch
size 128, default context length 10, hidden size 128, learning rate 0.01.
The batch size differs from the earlier batch-32 runs. These runs also change
input scope, training history, split, and test events; they cannot isolate a
single cause of performance differences from the historical results.

The current DeepCASE interpreter receives only zero training scores. Its output
is a diagnostic, not an unsupervised anomaly ranking. The legacy `score > 0`
classification is saved for transparency; negative/unscored outputs are counted
separately in metrics and retained in predictions. No AP is claimed. Vocabulary
is still constructed over the loaded interval and there is no validation partition.

Each epoch directory receives logs, configuration, split counts, raw predictions
with event IDs, test-event metadata, metrics, loss history, plots, and a context
builder checkpoint. `status.json` is authoritative for completion. A checkpoint
contains the context builder only, not a serialized interpreter.

Run from the repository root:

```bash
.venv/bin/python -u example/run_lanl_pilot.py \
  --output example/results/lanl/2026-09-24_remote-logon_pilot \
  --epochs 25 50 100
```

The runner refuses to overwrite experiment artifacts. To rerun, choose a new
output directory. Input preparation must finish successfully before training.
