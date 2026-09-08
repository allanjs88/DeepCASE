# Run the examples

Run all commands from the repository root.

## Setup (first time only)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install argformat numpy pandas scikit-learn scipy tqdm matplotlib
python -m pip install -e .
```

## Quick HDFS example

Requires `example/data/hdfs/hdfs_test_normal` (see [dataset instructions](../data/README.md)).

```bash
source .venv/bin/activate
OMP_NUM_THREADS=2 python example/example_hdfs.py --nrows 1000 --epochs 2 --no-show --output-dir saves/hdfs-quick
```

Plots are saved in `saves/hdfs-quick/`.

## Full HDFS run (100 epochs)

```bash
OMP_NUM_THREADS=2 python example/example_hdfs.py --no-show --output-dir saves/hdfs-full
```

Plots are saved in `saves/hdfs-full/`.

## Regular examples

Requires your own `data/example.csv` with `timestamp`, `event`, `machine`, and
`label` columns. Use actual labels, not all `-1`.

```bash
python example/example.py
python example/example_module.py
```

## LANL example

Requires the LANL `auth.txt` and `redteam.txt` files at the paths below.

```bash
python example/example_lanl.py --auth example/data/lanl/auth.txt --redteam example/data/lanl/redteam.txt
```
