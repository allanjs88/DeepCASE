# GPU epoch experiment results

All six independent runs completed on NVIDIA GeForce RTX 5070 using PyTorch 2.11.0+cu128. Random seeds: 42 (Python, NumPy, PyTorch). OMP_NUM_THREADS=2. Graphs generated using example/plotting.py.

HDFS: --nrows 1000; default training settings. The input limit counts sequences, producing 15,765 test events. HDFS evaluates next-event prediction.

LANL: auth_150000_154000.txt, redteam.txt, --nrows 0 --split-time 151800 --batch-size 32 --query-batch-size 128. 342,218 training events (3 redteam matches), 413,505 test events (2 redteam matches). LANL evaluates redteam classification using the example’s prediction > 0 rule.

| Dataset | Epochs | Accuracy | Macro F1 | Weighted F1 | Final training loss |
|---|---:|---:|---:|---:|---:|
| HDFS | 25 | 85.011101% | 0.411975 | 0.819362 | 0.035753 |
| HDFS | 50 | 85.829369% | 0.429950 | 0.830665 | 0.031271 |
| HDFS | 100 | 86.501744% | 0.451972 | 0.839430 | 0.027867 |
| LANL | 25 | 97.199550% | 0.493072 | 0.985794 | 0.052779 |
| LANL | 50 | 97.254689% | 0.493217 | 0.986078 | 0.050665 |
| LANL | 100 | 97.325304% | 0.493403 | 0.986440 | 0.048365 |

## LANL redteam detection

| Epochs | True positives | False positives | False negatives | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 25 | 2 | 11580 | 0 | 0.017268% | 100.00% |
| 50 | 2 | 11352 | 0 | 0.017615% | 100.00% |
| 100 | 2 | 11060 | 0 | 0.018080% | 100.00% |

Only two positive events occur in the test partition. High overall accuracy does not imply useful redteam precision; all models generated many false positives. The two datasets evaluate different tasks and their accuracies should not be directly compared.

## Files

Each dataset/epoch folder contains the prediction/confusion-matrix PNG, training-loss PNG, metrics.json, loss_history.json, predictions.npz, and run.log.

- HDFS 25 epochs: [predictions](hdfs/25/hdfs_prediction_results.png), [training loss](hdfs/25/hdfs_training_loss_summary.png), [metrics](hdfs/25/metrics.json).
- HDFS 50 epochs: [predictions](hdfs/50/hdfs_prediction_results.png), [training loss](hdfs/50/hdfs_training_loss_summary.png), [metrics](hdfs/50/metrics.json).
- HDFS 100 epochs: [predictions](hdfs/100/hdfs_prediction_results.png), [training loss](hdfs/100/hdfs_training_loss_summary.png), [metrics](hdfs/100/metrics.json).
- LANL 25 epochs: [predictions](lanl/25/lanl_prediction_results.png), [training loss](lanl/25/lanl_training_loss_summary.png), [metrics](lanl/25/metrics.json).
- LANL 50 epochs: [predictions](lanl/50/lanl_prediction_results.png), [training loss](lanl/50/lanl_training_loss_summary.png), [metrics](lanl/50/metrics.json).
- LANL 100 epochs: [predictions](lanl/100/lanl_prediction_results.png), [training loss](lanl/100/lanl_training_loss_summary.png), [metrics](lanl/100/metrics.json).

## Re-run

From the DeepCASE directory:

```bash
.venv/bin/python example/run_epoch_tests.py
```

The runner requires CUDA and refuses CPU fallback. Individual runs: `.venv/bin/python example/run_epoch_tests.py lanl 100` (dataset hdfs or lanl). Re-running overwrites the corresponding results. HDFS runs retain --no-graphs internally; the runner then generates plots from the results using ResultsPlotter. Original example scripts and plotting.py were not modified.
