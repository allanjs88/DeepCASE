# Remote LogOn pilot results — 2026-09-24

Independent CUDA runs using seed 42. Training: 321,368 events with no known
positives; test: 480,286 events with five known positives. Input extraction
scanned all 1,051,430,459 local source rows. The selected interval is
[144000, 154000), with train/test cutoff 147600. This uses only one hour of
pre-cutoff training history, not the complete first 41 hours.

| Epochs | Status | TP | FP | FN* | Unscored | Scored coverage | Recorded final loss | Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 25 | complete | 0 | 0 | 5 | 11,610 | 97.58% | 0.027999 | 170.5 |
| 50 | complete | 0 | 0 | 5 | 13,547 | 97.18% | 0.027766 | 306.8 |
| 100 | complete | 0 | 0 | 5 | 10,600 | 97.79% | 0.027549 | 593.5 |

*FN uses the inherited `prediction > 0` diagnostic: unscored events are
counted as negative. This is not a coverage-aware deployment decision rule.

The interpreter learns cluster scores from all-zero training labels, so it cannot
produce positive maliciousness scores in these runs. The zero detections expose
a scoring limitation; they do not establish that temporal representations or a
future GNN cannot detect the events. No unsupervised AP is claimed. Precision is
undefined when no positive alerts are issued; metrics.json uses zero_division=0.

Raw scores for the five known positives, in timestamp/event-row order:

- 25 epochs: 0.0, -3.0, -3.0, -3.0, -3.0.
- 50 epochs: 0.0, 0.0, -3.0, -3.0, -3.0.
- 100 epochs: -3.0, -3.0, -3.0, -3.0, -3.0.

Codes: 0 = matched benign-scored cluster; -1 = insufficient confidence;
-2 = event type absent from the interpreter training index; -3 = nearest cluster
farther than epsilon. Negative values are abstentions, not anomaly probabilities.

## Artifacts and checks

Each completed epoch folder contains `run.log`, `config.json`, `split_manifest.json`,
`metrics.json`, `loss_history.json`, `predictions.npz`, `test_events.csv.gz`,
`event_mapping.json`, `context_builder.pt`, and two PNG plots. Prediction arrays
are aligned across completed runs and checked for unique event IDs, label counts,
epoch counts, and agreement with saved metrics. The checkpoint stores only the
context builder. See [README.md](README.md) for protocol and reproduction.

## Comparison limits

The historical runs used another interval and split, all authentication rows,
batch size 32, and three known positives in training. This sweep uses remote
LogOn, pre-attack training, and batch size 128. These are not controlled before/
after accuracy comparisons. Loss uses the existing implementation normalization
and should not be compared numerically across different batch sizes.

Only one seed and five positive events are represented. Input vocabulary is still
factorized over the loaded interval; there is no validation partition. Training-only
encoding, validation, an unsupervised scorer, and broader temporal coverage remain
necessary before formal model comparison. Historical result folders were preserved.
