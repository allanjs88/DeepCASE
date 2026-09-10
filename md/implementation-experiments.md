# LANL implementation and experiment process

## Purpose and scope

Evaluate whether adding cross-host graph information to DeepCASE's GRU context
representations improves detection of known LANL compromise events. Compare the
DeepCASE baseline and GRU + GNN using the same data partitions, label access, and
evaluation protocol.

This document is a proposed implementation workflow, not a record of completed
experiments. Consult the [thesis schedule](schedule.md) for reported task status,
[commands](commands.md) for existing executable examples, and
[dataset documentation](../data/README.md#lanl-dataset) for LANL file details.
The steps below do not change the schedule's existing completion records.

Reference material:

- [Exploration notebook](/home/allanmurillo/Documents/src/deepcase/Datasets/DatasetLanl/exploration.ipynb), saved outputs reviewed on 2026-09-09; not rerun for this document.
- [Thesis proposal](/home/allanmurillo/Downloads/thesis-proposal.pdf), methodology and preliminary LANL analysis, especially sections 5.4–5.5 and Appendix A.
- [Current LANL example](../example/example_lanl.py).

The notebook and proposal are reference material. Operational choices below are
proposed working decisions unless explicitly attributed to those sources.

## Evidence motivating the process

| Observation from the saved notebook | Implementation or evaluation consequence |
| --- | --- |
| 20,000,000 sampled authentication rows, with 20 redteam matches | An always-negative predictor achieves 99.9999% accuracy. Report positive-class detection and false alerts. |
| 749 redteam records, spanning timestamps 150,885–2,557,047 | Count matched events in every partition; a large partition may still lack positives. |
| Authentication sample spans timestamps 1–5,011,199 | Full-period coverage does not establish complete temporal sequences. |
| Input is named `auth_sample_random.txt`; its generation is not recorded in the notebook | Establish sampling provenance. Random individual-row sampling can remove sequence context and graph interactions. |
| 321,268 records have different source and destination users | Test mismatch as an optional feature, not a label or mandatory filter. All 20 displayed positive rows have matching users. |
| Positive events connect source and destination hosts | Preserve both endpoints for graph modeling. Attack-only visualizations establish feasibility, not predictive performance. |

These counts describe the saved sample, not the full authentication dataset.

## 1. Record the environment and input provenance

- Record the repository commit, Python and dependency versions, operating system,
  GPU, PyTorch build, CUDA availability, and random seeds.
- Record source file paths, sizes, checksums, and whether files are compressed.
- Keep raw files unchanged. Store derived data separately with a configuration
  describing every selection and transformation.
- Verify a small CPU or CUDA run using [commands.md](commands.md). CUDA training
  does not remove the system RAM cost of loading and preparing data.

**Completion evidence:** an environment record and input manifest sufficient to
identify the exact data and software used by a run.

## 2. Implement reproducible data preparation

### Read, validate, and label

1. Read authentication data in chunks to avoid requiring the entire raw file in
   memory. Retain a stable event identifier, such as the original file row number.
2. Validate field counts and timestamp types. Preserve `?` as an explicit unknown
   category and report malformed records rather than silently dropping them.
3. Read redteam records and match the exact tuple
   `(timestamp, source_user, source_computer, destination_computer)`.
4. Assign `label=1` to matching authentication records and `label=0` otherwise.
   Here, zero means no known redteam match, not independently verified benign.
5. Report matched authentication rows, unique matched redteam keys, duplicate keys,
   and unmatched redteam keys separately. One key may match multiple auth rows;
   the two match counts need not be equal.
6. Preserve source/destination users, source/destination computers, authentication
   attributes, timestamp, event identifier, and label in the derived event table.

Use vectorized joins or tuple membership instead of the notebook's row-wise
`DataFrame.apply` when implementing the production pipeline.

### Select data without removing context

Use complete contiguous time intervals for primary experiments. Choose the initial
development interval using resource limits and coverage diagnostics, and record
its boundaries. Expand it after the pipeline works reliably.

Do not create the main evaluation dataset by retaining only attack rows or
randomly thinning individual authentication rows. If attack-centered windows are
used for debugging, label them as an enriched development dataset and do not
present their precision or prevalence as representative of the full period.

Sort chronologically with a deterministic tie-breaker. A row-number tie-breaker
provides reproducibility but does not establish physical ordering within a second.

**Completion evidence:** a reusable event table and a coverage report containing
rows, hosts, users, missing values, and positive counts by day or selected interval.

## 3. Freeze the temporal evaluation protocol

Define chronological training, validation, and test boundaries before model
comparison. Publish their exact timestamps and class counts. Select boundaries
using development considerations, not whichever test interval gives better scores.

The current example uses a 20%/80% row split after sorting and has no validation
partition. It is a smoke test, not the complete experiment protocol. A late test
interval may fall beyond the observed redteam period; report such intervals as
background-only evaluation, not attack-recall experiments.

Prevent information from crossing the split incorrectly:

- Fit categorical vocabularies, normalization, and learned parameters on training
  data. Define an unknown category for unseen validation/test event types.
- Use validation data for thresholds and hyperparameters. Freeze these before
  test evaluation.
- Allow only preceding observations in a target event's temporal context. Earlier
  training history may initialize validation/test context if this policy is shared
  by both models and does not update learned parameters or use held-out labels.
- Keep simultaneous timestamps on the same side of a partition boundary.
- Define whether scoring happens at event arrival or after a completed window.
  For the initial protocol, use event arrival: graph context and host embeddings
  use strictly earlier timestamps; current event attributes may be scored directly.
  Do not aggregate later events from the target's window into its features.
- Keep evaluation prevalence intact. Apply any resampling or class weighting only
  during training and record it.

**Completion evidence:** a saved split manifest and checks for temporal ordering,
label coverage, stable event alignment, and absence of future information.

## 4. Establish the DeepCASE baseline

Start from the existing example and document its behavior:

- Source computer becomes `machine`, defining host-local sequences.
- Default event categories combine authentication type, logon type, orientation,
  success, and `user_mismatch`.
- Destination computer is not retained in the final DeepCASE input table.
- The example factorizes categories over the entire loaded sample; replace this
  with the training vocabulary policy for formal experiments.
- Training labels are supplied to the interpreter through `scores=labels_train`.
  Account for this supervision when comparing against a supervised GNN.
- The example converts predictions using `prediction > 0`. Inspect interpreter
  output semantics, including unknown/unscored outcomes, before treating outputs
  as probabilities or using them for ranking metrics.

Save raw outputs, event identifiers, and labels. Report interpreter coverage and
define how unscored events are handled; do not silently count abstentions as
benign. Fix the non-writable NumPy label-conversion warning with a writable copy
and verify that label values and alignment remain unchanged.

**Completion evidence:** a reproducible baseline run with class counts, confusion
matrix, output interpretation, runtime, and peak memory usage.

## 5. Build causal host graphs and integrate the GRU

Construct directed host graphs from all authentication activity in the selected
history, including unmatched events. Redteam labels must not determine whether
an edge exists or enter node/edge features.

| Component | Proposed initial definition |
| --- | --- |
| Nodes | Source and destination computers present in the available history |
| Edges | Directed source-to-destination authentication relationships |
| Repeated edges | Aggregate within the history window; retain count, success ratio, and recency |
| Node features | GRU representation of preceding host-local events; optional historical activity statistics |
| Event prediction | Source embedding, destination embedding, and current authentication attributes |
| New hosts | Explicit initialization policy when historical context is unavailable |

Keep self-authentication events initially and document their treatment separately
from GNN self-loops. Do not infer a chronological attack path from a static graph.
User identity may support historical statistics, but `user_mismatch` alone does
not encode credential reuse across hosts.

Implement one directed message-passing architecture first. Start with a frozen
GRU encoder to verify integration, then treat joint fine-tuning as a separate
experiment. Document pooling from event contexts to host representations, tensor
shapes, history duration, edge aggregation, and scoring/loss functions.

**Completion evidence:** a small end-to-end run and meaningful checks that edge
direction is preserved, repeated events aggregate correctly, embeddings align with
hosts, gradients reach intended components, and future observations cannot change
an earlier prediction.

## 6. Run controlled experiments

The following matrix is proposed, not a list of completed runs. Use the same
partitions, scored events, label budget, and comparable tuning effort throughout.

| ID | Configuration | Question |
| --- | --- | --- |
| E0 | Always predict unmatched/negative | How misleading is accuracy under the observed imbalance? |
| E1 | DeepCASE with the documented LANL event encoding | What does the baseline achieve? |
| E2 | GRU features with the proposed event scoring head, without message passing | Does changing the scorer explain improvement? |
| E3 | GRU + GNN with the same scoring head as E2 | Does relational information add value? |
| E4 | E3 with and without `user_mismatch` | Does this feature help beyond temporal and relational context? |
| E5 | E3 across a small predefined history-window grid | How sensitive are detection and resource use to graph history? |

Use development pilots to choose feasible settings. Freeze the final matrix and
run at least five paired seeds as a proposed starting point, matching the same
seed list across configurations. Repeated seeds measure optimization variability;
they do not create independent attack campaigns or datasets.

Keep successful and failed run records. Avoid changing the test protocol after
seeing results; classify later exploratory changes separately.

## 7. Evaluate detection and operational cost

Report positive-class precision, recall, F1, false positive rate, confusion counts,
and the number of known positive events. Include runtime, peak RAM/VRAM, and
false alerts per day where the selected intervals support that denominator.

For ranked evaluation, define the score first. Average precision and trapezoidal
PR-AUC are different summaries; name the calculation used and apply it consistently.
Do not calculate a claimed continuous-score PR curve from thresholded class labels.
If the baseline only exposes discrete interpreter scores, disclose the limited
ranking resolution and retain a fair thresholded comparison.

Tune thresholds on validation data, preferably against a predefined false-positive
budget. The proposal's improvement criterion is higher recall and AUC-PR with an
equal or lower false positive rate. Report tradeoffs when that criterion is not met.

For partitions with no known positives, mark attack recall and PR summaries as
not applicable. With very few positives, show absolute detected/missed counts and
avoid strong generalization claims.

Compare paired runs and report variability. For uncertainty from data sampling,
use temporal blocks or attack-group units justified by the data; individual auth
rows are correlated. Explain the resampling unit and limits of any significance
test. Inspect errors by time, host, authentication type, and previously unseen
entities to check for reliance on a frequently compromised host identifier.

**Completion evidence:** comparison tables, precision-recall plots where valid,
resource measurements, and an error analysis tied to saved predictions.

## 8. Save artifacts and update thesis evidence

Suggested output structure; these directories are not created by this document:

```text
saves/lanl/<experiment-id>/<seed>/
  config.json
  environment.txt
  data_manifest.json
  split_manifest.json
  metrics.json
  predictions.csv
  training.log
  model/
  figures/
```

Record the commit, input checksums, time boundaries, mappings, seed, feature set,
supervision policy, model parameters, threshold, and unscored-event policy. Include
stable event identifiers, timestamps, labels, raw scores, and predicted classes in
prediction artifacts.

Link verified artifacts into the [schedule session log](schedule.md#session-log)
when implementation or experiments are completed. Separate observed results from
interpretations and proposed changes in the thesis write-up.

## Immediate next deliverable

Implement a chunked LANL coverage report showing authentication rows, unique
redteam matches, positive prevalence, and host counts by day. Use it to specify a
contiguous development interval and explicit temporal partitions before running
the formal DeepCASE baseline. An interval extractor is now available as
[`example/prepare_lanl_interval.py`](../example/prepare_lanl_interval.py), with
[usage commands](commands.md#prepare-a-contiguous-lanl-development-interval). It
reports exact training/test matches for a specified interval. The full daily
coverage report and GNN implementation remain separate next steps.
