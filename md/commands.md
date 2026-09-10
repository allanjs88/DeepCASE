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

### Enable NVIDIA CUDA (optional)

The setup above installs CPU-only PyTorch. `nvidia-smi` confirms that the NVIDIA
driver sees the GPU, but PyTorch also needs a CUDA-enabled build. For the RTX 5070,
install the CUDA 12.8 build in the activated virtual environment:

```bash
source .venv/bin/activate
python -m pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

This version is listed in the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
`requirements.txt` currently pins `torch==2.14.0`; installing those requirements
afterward overrides this version. Run the CUDA installation command after other
dependency installation steps when using this configuration.

Check the installed build and CUDA detection:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

A version ending in `+cpu` with `CUDA build: None` indicates CPU-only PyTorch.
Once CUDA is available, verify an actual GPU tensor allocation:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

The LANL example automatically selects CUDA when `torch.cuda.is_available()` is
true; rerun it after installing the CUDA build.

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

The default sample contains the first 100,000 authentication rows and may have
zero redteam matches. A report showing 100% accuracy for class `0` alone does not
demonstrate attack detection. Inspect a larger sample without training:

```bash
python example/example_lanl.py --auth example/data/lanl/auth.txt --redteam example/data/lanl/redteam.txt --nrows 1000000 --summary --skip-model
```
To read all the file
```bash
python example/example_lanl.py \
  --auth example/data/lanl/auth.txt \
  --redteam example/data/lanl/redteam.txt \
  --nrows 0 --epochs 50
```

Improve query

```bash
python example/example_lanl.py \
  --auth example/data/lanl/auth.txt \
  --redteam example/data/lanl/redteam.txt \
  --epochs 50 \
  --nrows 100000 \
  --batch-size 32 \
  --query-batch-size 128
```

```bash
python example/example_lanl.py \
  --auth example/data/lanl/auth_150000_154000.txt \
  --redteam example/data/lanl/redteam.txt \
  --nrows 0 \
  --split-time 151800 \
  --epochs 20 \
  --batch-size 32 \
  --query-batch-size 128
```

Increase `--nrows` as needed; `--nrows 0` reads the full authentication file into
memory and requires substantially more RAM. Remove `--skip-model` to train and
evaluate. See the [LANL dataset documentation](../data/README.md#lanl-dataset)
for download instructions and label interpretation.

The warning about a non-writable NumPy array during label conversion is separate
from CUDA detection; changing the PyTorch build does not address that warning.
