# Examples
Here we list three examples of how to use DeepCASE.

## Regular usage
As explained in the [documentation](deepcase.readthedocs.io), DeepCASE offers two interfaces:
 1. The `DeepCASE` approach as described in the paper, with each individual step as a separate method. This includes methods where clusters can be manually labelled in a separate step. This is implemented in `example.py`
 2. A `DeepCASE` module that can be used to `fit` and `predict` samples using a single `fit_predict` method. Note that this method only works if we already have labelled the sequences we input into DeepCASE. We show an example of using this interface in `example_module.py`.

## Context Builder Sequence Prediction
Besides using the entire workflow, we can also use DeepCASE's ContextBuilder to predict the next item in a sequence.
`example_hdfs.py` gives an example on how to use the DeepCASE's ContextBuilder to predict the next item in the HDFS dataset (Table IV in paper).

## LANL authentication dataset
`example_lanl.py` reads the LANL `auth.txt`/`auth.txt.gz` and `redteam.txt`/`redteam.txt.gz` files, converts authentication rows into DeepCASE sequences, and uses exact redteam matches as malicious labels.

The full LANL authentication file is large, so the example reads 100,000 rows by default:

```bash
python example/example_lanl.py --auth example/data/lanl/auth.txt --redteam example/data/lanl/redteam.txt
```

Use `--nrows 0` for the full file, `--skip-model` when you only want to verify preprocessing, or `--event-fields` to choose which auth columns define the discrete event type.

## Paired LANL processing pilot

From the `DeepCASE` repository directory, with the existing environment activated:

```bash
python example/compare_lanl.py --plan --output /tmp/lanl-plan
python -u example/compare_lanl.py --epochs 3 --device cuda \
  --output "example/results/lanl-comparison-$(date +%Y%m%d-%H%M%S)"
```

Requires `requirements-gnn.txt` and the existing
`example/data/lanl/auth_144000_154000.txt` and `redteam.txt` files. The plan reads
data and prints counts without training or writing files. Output directories must
be new. Use `--device cpu` if CUDA is unavailable.

Defaults select the complete interval [150820,150910), split into three 30-second
partitions, with a 30-second history window. This attack-centered interval is a
processing smoke test, not an independently selected performance benchmark.
Both implementations receive the same train/test events and train-only event
vocabulary. Histories exclude simultaneous and future events. Original DeepCASE
uses its actual context builder and interpreter through the Python API, with a
shared-protocol sequence adapter instead of CLI preprocessing. Validation events
provide historical context for both; only the GNN calibrates on validation.

Outputs include shared partition CSVs and vocabulary, per-model checkpoints,
predictions, losses, timings and reports, plus `comparison.csv`, `comparison.json`
and `status.json`. Original negative interpreter codes remain explicit abstentions:
scored false negatives and unscored positives are separate, and total attacks not
alerted includes both. Training losses have different definitions and are not
directly comparable. Original benign-only cluster labels cannot produce positive
attack scores; this pilot does not establish a fair unsupervised detection baseline.

## Run the LANL models independently

The paired runner is also available as two separate entry points. Each accepts
all the same arguments and defaults as `compare_lanl.py`, including `--plan`:

```bash
python -u example/experiment_deepcase_lanl.py --epochs 3 --device cuda \
  --output "example/results/lanl-original-$(date +%Y%m%d-%H%M%S)"

python -u example/experiment_deepcase_gnn_lanl.py --epochs 3 --device cuda \
  --output "example/results/lanl-gnn-$(date +%Y%m%d-%H%M%S)"
```

Each command runs only its named model. Keep interval, window, epoch and seed
arguments equal when comparing independent runs. Shared preparation and reporting
live in `lanl_experiment.py`; each experiment script contains its model-specific
training and evaluation. `compare_lanl.py` still runs both in one invocation.

Each output root contains the manifest, partition CSVs, vocabulary, status, and
single-model `comparison.json`/`comparison.csv`. Model artifacts are under
`original/` for DeepCASE or `gnn/` for DeepCASE-GNN, including `report.json`,
`predictions.csv`, `loss_history.json`, and `model.pt`. Existing directories are
rejected to protect previous results. These are the same processing diagnostics
and scoring protocols as the paired runner, not a new detection methodology.

## Experiment graphs and confusion reports

Both independent LANL scripts (and the paired runner) automatically save a
`graphs/` folder under each model's output directory:

- `confusion_matrix.png` and `confusion_matrix.csv`: actual benign/attack rows,
  predicted benign/attack/unscored columns; abstentions remain separate.
- `confusion_report.txt`: text report using a copy of the original utility.
- `training_loss.png`: the model's loss by epoch.
- `processing_time.png`: recorded stage durations, excluding reporting time.

Shared plotting code is in `example/reporting/`. `example/plotting.py` retains a
compatibility import for existing examples. The `deepcase` package is unchanged.
To generate or refresh graphs from an existing run without training again:

```bash
python -m example.reporting example/results/lanl-original-20261010-081750/original
python -m example.reporting example/results/lanl-gnn-20261010-081812/gnn
```
