# LANL contiguous interval preparation

Date: 2026-09-09

## Purpose

Prepare a manageable, contiguous authentication interval with verified redteam
matches in both training and testing. The previous 100,000-row example contained
no known positives, so its 100% accuracy did not measure attack detection.

This work prepared and verified data. It did not train a model or establish
detection performance.

## Changes implemented

- Added [prepare_lanl_interval.py](../example/prepare_lanl_interval.py), which
  streams authentication records to an interval extract without loading the full
  source into RAM. It supports plain text and gzip inputs, scans to EOF without
  assuming chronological source order, and refuses to overwrite an existing extract.
- Added exact-match coverage reporting: authentication rows, positive rows,
  unique matched redteam keys, and unmatched redteam keys for each partition.
  A JSON report is written alongside the extract after the scan completes.
- Updated [example_lanl.py](../example/example_lanl.py) with `--split-time` and
  `--report-only`. It prints partition sizes, timestamp ranges, and positive counts
  before allocating context tensors. Report-only mode then exits without training.
- Made initial authentication sorting stable. Explicit timestamp splits keep equal
  timestamps in the same partition. The default ratio-based split now chooses a
  timestamp boundary near the requested ratio, so actual proportions can differ.
- Added [targeted tests](../tests/test_lanl_interval.py) and updated
  [commands.md](commands.md#prepare-a-contiguous-lanl-development-interval) and
  [the implementation process](implementation-experiments.md).

## Interval selection and labeling

The local redteam file contains five records in `[150000, 154000)` seconds.
This interval was selected to exercise positive labels on both sides of a split
at timestamp `151800`.

Training uses `150000 <= timestamp < 151800`. Testing uses
`151800 <= timestamp < 154000`. The boundaries are relative LANL timestamps,
not calendar dates.

The extract includes every authentication row in the interval, including all
hosts and unmatched activity. It is not an attack-only selection. Labels match
the exact tuple:

```text
(timestamp, source_user, source_computer, destination_computer)
```

A matched authentication row receives label `1`. Label `0` means no matching
redteam record, not independently verified benign activity. Positive-row counts
and unique matched keys are tracked separately because duplicate authentication
keys can otherwise obscure label coverage.

## Verified results

The extractor scanned all **1,051,430,459 authentication rows** in the local
approximately 69 GB source file and retained **755,723 rows**.

| Partition | Observed timestamp range, inclusive | Authentication rows | Positive rows | Unique matched redteam keys | Unmatched redteam keys in interval |
| --- | --- | --- | --- | --- | --- |
| Training | 150000–151799 | 342,218 | 3 | 3 | 0 |
| Testing | 151800–153999 | 413,505 | 2 | 2 | 0 |
| Total | 150000–153999 | 755,723 | 5 | 5 | 0 |

The completed extract was independently loaded through the example's report-only
mode. Its partition counts agreed with the streaming report, and its default
encoding produced **67 distinct DeepCASE event categories**.

Local artifacts:

- Authentication extract: `example/data/lanl/auth_150000_154000.txt`.
- Coverage report: `example/data/lanl/auth_150000_154000.txt.report.json`.
- Original inputs: `example/data/lanl/auth.txt` and `example/data/lanl/redteam.txt`.

These data artifacts are under the repository's ignored `example/data/` directory.
The JSON report records source paths, boundaries, and counts; it does not currently
record input checksums or a complete environment manifest.

## Reproduce the preparation

Run from the repository root with the virtual environment activated:

```bash
source .venv/bin/activate
python example/prepare_lanl_interval.py \
  --auth example/data/lanl/auth.txt \
  --redteam example/data/lanl/redteam.txt \
  --output example/data/lanl/auth_150000_154000.txt \
  --start 150000 --end 154000 --split-time 151800
```

The extract already exists locally; reuse it rather than repeating the full scan.
To regenerate independently, supply a new output path. Wait for successful
completion and the JSON report before reading the output: the extractor writes
directly to its destination, and an interrupted run can leave a partial file.

Verify the completed extract without training:

```bash
python example/example_lanl.py \
  --auth example/data/lanl/auth_150000_154000.txt \
  --redteam example/data/lanl/redteam.txt \
  --nrows 0 --split-time 151800 --report-only
```

`--nrows 0` reads the complete extracted file. Omitting it would apply the default
100,000-row cap and truncate the intended interval. Report-only mode avoids
context tensors but still loads and encodes the extracted data in RAM.

## Checks performed

Two targeted unit tests passed:

1. Complete extraction from an unsorted gzip input, inclusive/exclusive interval
   boundaries, exact user matching, duplicate positive rows versus unique keys,
   unmatched redteam keys, and refusal to overwrite an existing extract.
2. Explicit and ratio-derived timestamp splits keep ties together and reject an
   empty partition.

```bash
python -m unittest discover -s tests -p test_lanl_interval.py
```

The real-data report-only command completed successfully and confirmed the counts
above. `git diff --check` also passed. No model-training run was performed as part
of this preparation.

## Next experiment and limits

Start with a short pilot run:

```bash
python example/example_lanl.py \
  --auth example/data/lanl/auth_150000_154000.txt \
  --redteam example/data/lanl/redteam.txt \
  --nrows 0 --split-time 151800 --epochs 3 \
  --batch-size 32 --query-batch-size 128
```

Smaller batches reduce working memory, but the example still moves the complete
context and event tensors to the selected device. This command is a next step,
not a recorded successful training result.

The pilot has only two known positive test events; detecting or missing one changes
attack recall by 50 percentage points. Its attack-informed interval selection is
useful for development, not representative final evaluation. History before
timestamp `150000` is absent, so initial contexts are incomplete.

The example also retains sample-wide event encoding and has no validation
partition. Its binary conversion treats scores at or below zero as class `0`,
including any unscored negative outputs. Formal experiments still need the
training-only vocabulary, validation protocol, explicit unscored-event handling,
broader temporal coverage, and repeated comparisons described in
[implementation-experiments.md](implementation-experiments.md).
