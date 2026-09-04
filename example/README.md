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
