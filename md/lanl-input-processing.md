# LANL example input processing

`example/example_lanl.py` follows the proposed input scope in
`lanl-larroche-paper.md`: LogOn events between distinct known source and destination
hosts, with all authentication mechanisms, account categories, and success/failure
outcomes retained. Unknown hosts (`?` or empty) cannot establish a remote relationship
and are excluded. No user selection, benign downsampling, or duplicate merging occurs.
Labels match the original four-field Red Team keys exactly.

The default cutoff is 147600 seconds (41 hours): earlier rows train, later rows test.
Equal timestamps stay together. `--split-time` overrides the cutoff;
`--train-ratio 0.2` explicitly selects a chronological pilot split.
`--auth-scope all` restores the original unfiltered authentication scope.

For a small pipeline check, from the project directory:

```bash
python example/example_lanl.py --train-ratio 0.2 --report-only
```

The default 100000 physical-row prefix is only a pilot and may not span the default
cutoff. For a complete interval, use `example/prepare_lanl_interval.py`, then pass
its output using `--auth PATH --nrows 0`. The example loads its input into memory;
it is not a streaming full-dataset training implementation.

The returned frame retains original authentication fields and an `event_id` (zero-based
physical row in the supplied file), alongside DeepCASE's required columns. This allows
future graph construction to preserve direction and map results back to individual rows.
IDs from different extracts are not global dataset IDs.

This is an input-processing update, not a reproduction of the paper. Event vocabulary
is still factorized over the loaded input, including test data. A strict inductive
experiment needs a training-only vocabulary with explicit unknown-event handling.
The current DeepCASE interpreter uses cluster labels, not an unsupervised anomaly
score; with benign-only training, its classification report is only a diagnostic.
Model fitting rejects known Red Team events in training. A paper-comparable experiment
still requires unsupervised scoring and event-level precision/recall, FPR, and AP.
