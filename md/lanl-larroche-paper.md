# Methodological Considerations from *On Fair and Realistic Performance Evaluations for Graph-Based Lateral Movement Detectors*

## 1. Why This Paper Is Important for the Thesis

The paper **"On Fair and Realistic Performance Evaluations for Graph-Based Lateral Movement Detectors"** by Corentin Larroche is particularly relevant to the proposed extension of DeepCASE using GRU and Graph Neural Networks (GNNs).

The main contribution of the paper is not a new lateral movement detection architecture. Instead, it investigates how **preprocessing, graph construction, train/test splitting, labeling, and evaluation decisions can significantly affect the reported performance of graph-based lateral movement detectors**.

This is directly relevant to experiments using the **Los Alamos National Laboratory (LANL) Comprehensive, Multi-Source Cyber-Security Events dataset**.

The central methodological warning can be summarized as:

> Different preprocessing and labeling decisions can produce substantially different performance results, even when evaluating the same detection method.

Therefore, the comparison between:

```text
DeepCASE
    vs.
DeepCASE + GRU/GNN
```

must use exactly the same underlying data, preprocessing rules, labels, and evaluation methodology.

Otherwise, an apparent improvement could result from preprocessing differences rather than from the proposed GNN architecture.

---

# 2. Graph Representation

The paper describes three major ways of constructing graphs from authentication or network-flow events.

## 2.1 Static Graph

All events are aggregated into a single graph.

```text
All events
    │
    ▼

    C2
   ↗  ↘
 C1    C4
   ↘  ↗
    C3
```

This representation is simple, but temporal information can be lost.

---

## 2.2 Discrete-Time Graph

Events are divided into temporal windows of size:

\[
\Delta
\]

A separate graph is constructed for each window:

\[
G_1,G_2,G_3,\ldots,G_T
\]

For example:

```text
Authentication events
        │
        ▼

0 ───── 300 ───── 600 ───── 900 seconds
│         │          │
▼         ▼          ▼

G1        G2         G3
```

Each graph can be represented as:

\[
G_t=(V_t,E_t,X_t,A_t)
\]

where:

- \(V_t\) = active hosts
- \(E_t\) = authentication relationships
- \(X_t\) = node features
- \(A_t\) = edge attributes

This representation closely matches the current proposed GRU + GNN architecture.

---

## 2.3 Continuous-Time Graph

Instead of grouping events into snapshots, the exact timestamp is preserved for every edge.

For example:

```text
C1 ──t=100──→ C2
C2 ──t=105──→ C3
C3 ──t=112──→ C4
```

Conceptually:

\[
e_i=(u_i,v_i,t_i)
\]

where:

- \(u_i\) = source host
- \(v_i\) = destination host
- \(t_i\) = exact timestamp

This representation preserves more temporal information but generally increases model complexity.

---

# 3. Graph Representation Proposed for the Thesis

The proposed architecture can use a **discrete-time temporal graph**.

Authentication events occurring during temporal window \(t\) produce:

\[
G_t=(V_t,E_t,X_t,A_t)
\]

For example:

```text
Window t

C10 ─────────→ C20
 │              │
 │              ▼
 └───────────→ C30
```

A directed edge:

\[
(i,j)\in E_t
\]

exists when an authentication event from host \(i\) to host \(j\) occurs during temporal window \(t\).

This representation preserves cross-host relationships while allowing the system to process the LANL dataset as a sequence of graph snapshots.

---

# 4. Do Not Filter Only NTLM Events

One important recommendation from the paper is to avoid restricting the LANL dataset to only NTLM authentication events.

Some previous lateral movement detection studies used only:

```text
NTLM + Remote LogOn
```

However, lateral movement can also use other authentication mechanisms such as:

```text
Kerberos
Negotiate
```

The LANL Red Team events happen to use NTLM. Therefore, selecting only NTLM events removes many benign authentication events while retaining the known malicious events.

This makes the detection problem artificially easier.

## Avoid

```python
df = df[df["auth_type"] == "NTLM"]
```

## Prefer

Keep the different authentication mechanisms:

```text
C10 ──Kerberos───→ C20

C30 ──NTLM───────→ C40

C50 ──Negotiate──→ C60
```

Authentication type can instead become an **edge feature**.

For example:

\[
e_{ij}
=
(
source,
destination,
authentication\ type,
timestamp
)
\]

---

# 5. Keep Computer Accounts

LANL contains machine accounts such as:

```text
C553$@DOM1
```

The `$` indicates a Windows computer/machine account.

It may be tempting to remove these records because they do not represent normal human users.

However, the paper recommends against automatically removing them.

Computer accounts can authenticate remotely and may have access privileges that make them relevant for lateral movement.

Therefore:

```text
C553$@DOM1
```

should not automatically be considered noise.

Instead, account type could potentially become a feature:

```text
account_type = USER

or

account_type = COMPUTER
```

For example:

```text
C553$@DOM1
     │
     ▼

C553 ─────────────→ C850
     authentication
```

---

# 6. Keep Local and Built-In Accounts

The paper also recommends against automatically removing:

```text
Local accounts
Built-in accounts
Computer accounts
```

Built-in accounts may participate in legitimate administrative activity but can also be used by lateral movement techniques.

For example, tools such as PsExec may involve built-in system accounts when executing commands remotely.

Therefore, removing these accounts could eliminate behavior that is relevant to lateral movement detection.

---

# 7. Do Not Downsample Benign Users for Evaluation

One particularly problematic preprocessing strategy is removing users that are not associated with Red Team activity.

For example:

```text
Original dataset

U1    benign
U2    benign
U3    benign
U4    benign
U5    Red Team
U6    benign
...
```

If most benign users are removed:

```text
U5    Red Team
U6    benign
```

the detection problem becomes substantially easier.

The number of false-positive opportunities decreases.

This can artificially improve:

\[
Precision
\]

\[
FPR
\]

\[
AUC
\]

and other metrics.

In a real deployment, defenders do not know beforehand which users will become compromised.

Therefore:

> Benign users should not be removed based on knowledge of Red Team activity.

---

# 8. Strict Temporal Train/Test Separation

The paper emphasizes that training and testing should not overlap temporally.

A real-world detector can only learn from:

\[
\text{past data}
\]

and then detect anomalies in:

\[
\text{future data}
\]

Therefore:

```text
TRAIN              TEST
───────────────│─────────────────────────────→ time
               ↑
          temporal split
```

is preferable to:

```text
random train/test split
```

---

# 9. Recommended LANL Split

For the LANL dataset, the paper recommends:

```text
Training = first 41 hours
Testing  = remainder of the dataset
```

Conceptually:

```text
LANL timeline

0                  41 hours                         58 days
│────────────────────│────────────────────────────────│
        TRAIN                         TEST

      benign                   benign + Red Team
```

The motivation is that:

1. Training occurs before the first known malicious actions.
2. There is no temporal overlap between training and testing.
3. Training represents past behavior.
4. Testing represents future behavior.
5. Keeping the later benign period provides a better estimate of false positives.

This split should therefore be seriously considered for the primary LANL experiment.

---

# 10. Why a Random 70/15/15 Split Could Be Problematic

A traditional machine-learning split such as:

```text
70% Train
15% Validation
15% Test
```

can be problematic if events are randomly assigned.

For example:

```text
Day 40 event → TRAIN
Day 5 event  → TEST
Day 20 event → TRAIN
Day 10 event → TEST
```

The model would effectively learn from the future to predict the past.

This creates temporal leakage.

Therefore, if train/validation/test partitions are required, they should remain chronological:

```text
Past                           Future
│                                 │
▼                                 ▼

TRAIN ─────── VALIDATION ─────── TEST
```

rather than randomly sampling events.

---

# 11. Snapshot Size Is an Experimental Variable

For discrete-time graphs, temporal window size is:

\[
\Delta
\]

For example:

\[
\Delta=300\text{ seconds}
\]

would produce 5-minute graph snapshots.

Other possibilities include:

\[
\Delta=600
\]

\[
\Delta=1800
\]

\[
\Delta=3600
\]

corresponding approximately to:

```text
5 minutes
10 minutes
30 minutes
1 hour
```

The paper demonstrates that snapshot size can significantly affect evaluation.

Therefore, snapshot size should not be selected arbitrarily.

It should either:

1. be justified theoretically,
2. be determined experimentally using validation data, or
3. be evaluated as an experimental factor.

---

# 12. Snapshot Size Changes the Graph

Consider these events:

```text
100  C1 → C2
110  C1 → C2
120  C1 → C2
130  C1 → C2
```

With a sufficiently large snapshot, they may become a single graph edge:

```text
C1 ═══════→ C2

weight = 4
```

Instead of four independent events:

```text
C1 → C2
C1 → C2
C1 → C2
C1 → C2
```

Therefore, changing:

\[
\Delta
\]

changes:

- number of graph edges,
- edge weights,
- graph density,
- number of evaluation samples,
- class imbalance.

This is an important methodological consideration.

---

# 13. Edge Aggregation

Suppose one temporal window contains:

```text
100,C1,C2,benign
110,C1,C2,benign
120,C1,C2,benign
130,C1,C2,malicious
```

Graph construction may aggregate these events into:

```text
C1 ─────────→ C2
```

with:

\[
weight(C1,C2)=4
\]

However, now the edge contains both:

```text
3 benign events
1 malicious event
```

This creates an important question:

> Should the aggregated edge be considered benign or malicious?

This illustrates why graph construction and evaluation should be treated separately.

---

# 14. Evaluate at the Event Level

One of the strongest recommendations from the paper is to evaluate competing detectors using the same underlying **events**, rather than allowing graph aggregation to redefine the evaluation samples.

A useful principle for the thesis is:

> **Graph construction can aggregate information; evaluation should preserve event-level ground truth.**

For example:

```text
Original authentication events

E1: C1 → C2 benign
E2: C1 → C2 benign
E3: C1 → C2 benign
E4: C1 → C2 malicious
```

may become:

```text
Graph snapshot

C1 ═══════→ C2
```

for message passing.

However, predictions should eventually be mapped back to:

```text
E1
E2
E3
E4
```

for evaluation whenever the architecture permits it.

This makes comparisons between models more consistent.

---

# 15. Use Original LANL Red Team Labels

LANL provides known Red Team authentication events.

The paper recommends using these original labels rather than attempting to infer additional malicious events.

For example, avoid creating rules such as:

```python
if source_host == "C17693":
    malicious = True
```

or:

```python
if user == "U748" and timestamp_near_attack:
    malicious = True
```

These rules would introduce labels that LANL itself does not provide.

Instead:

```text
auth.txt
    │
    │ match
    ▼
redteam.txt
    │
    ▼
Ground truth
```

should be used.

---

# 16. Important Limitation of LANL Labels

The paper emphasizes that the official LANL description does not explicitly state that every labeled Red Team authentication represents lateral movement.

Furthermore, the labels represent only a subset of all activity associated with the Red Team exercise.

However, the dataset does not provide enough information to reliably reconstruct all malicious events or determine their exact high-level attack tactic.

Therefore, the paper argues that using the original LANL labels is the most defensible evaluation strategy despite their limitations.

This limitation should be explicitly documented in the thesis.

---

# 17. The C17693 Problem

One of the most important LANL limitations identified by the paper is that almost all labeled malicious authentication events originate from:

```text
C17693
```

Furthermore, all labeled malicious authentications belong to a **single simulated intrusion**.

Conceptually:

```text
                         ┌────→ C728
                         │
                         ├────→ C1173
                         │
C17693 ──────────────────┼────→ C294
                         │
                         ├────→ C...
                         │
                         └────→ C...
```

This has significant implications.

The experiment should not be interpreted as demonstrating that:

> The proposed model generalizes to every possible enterprise lateral movement attack.

A more defensible interpretation is:

> The experiment evaluates the model's ability to detect the labeled malicious authentication events associated with the LANL Red Team intrusion.

This limitation should appear explicitly in the thesis.

---

# 18. Avoid Learning Directly from LANL Attack Labels

Because the labeled malicious events largely belong to a single simulated intrusion, using part of that intrusion for supervised training and the rest for testing can produce an unrealistic experiment.

For example:

```text
Single intrusion

Attack phase 1 ───→ Attack phase 2 ───→ Attack phase 3
      │                                      │
      ▼                                      ▼
    TRAIN                                   TEST
```

The model may simply learn characteristics of the same intrusion that it is later asked to detect.

The paper therefore recommends using LANL primarily for **unsupervised detection**, unless supervised training is performed using another dataset.

This is compatible with the anomaly-detection nature of DeepCASE.

---

# 19. Extreme Class Imbalance

Under the preprocessing proposed by the paper, the LANL evaluation data contains approximately:

```text
Nodes             = 16,159
Events            = 369,569,626
Malicious events  = 702
```

The proportion of malicious events is approximately:

\[
\frac{702}{369,569,626}
\approx
1.9\times10^{-6}
\]

or approximately:

\[
0.00019\%
\]

This represents extreme class imbalance.

Conceptually:

```text
Benign events:

██████████████████████████████████████████████████

Malicious:

.
```

---

# 20. Why Accuracy Is Not Appropriate

Suppose there were:

```text
999,999 benign events
1 malicious event
```

A detector predicting:

```text
EVERYTHING = BENIGN
```

would obtain:

\[
Accuracy=
\frac{999999}{1000000}
=
99.9999\%
\]

despite detecting:

\[
0
\]

attacks.

Therefore, accuracy should not be a primary evaluation metric for LANL lateral movement detection.

---

# 21. False Positive Rate Is Important

False Positive Rate is:

\[
FPR=
\frac{FP}{FP+TN}
\]

Because LANL contains an enormous number of benign events, even a very small FPR can produce many false alerts.

For example, suppose:

\[
10,000,000
\]

benign events are evaluated with:

\[
FPR=0.1\%
\]

Then:

\[
FP=
10,000,000\times0.001
\]

\[
FP=10,000
\]

That means:

```text
10,000 false alerts
```

despite an apparently low FPR.

This is highly relevant in a Security Operations Center because analysts ultimately need to investigate the generated alerts.

---

# 22. Precision Is Important

Precision is:

\[
Precision=
\frac{TP}{TP+FP}
\]

It answers:

> Of all events identified as suspicious, how many were actually malicious?

For example:

```text
100 alerts generated

10 = true malicious events
90 = false positives
```

Then:

\[
Precision=
\frac{10}{100}
=
0.10
\]

or:

\[
10\%
\]

This metric is particularly important when malicious events are extremely rare.

---

# 23. Recall Is Important

Recall is:

\[
Recall=
\frac{TP}{TP+FN}
\]

It answers:

> Of all known malicious events, how many did the detector identify?

For example:

```text
100 known attacks

80 detected
20 missed
```

Then:

\[
Recall=
\frac{80}{100}
=
0.80
\]

or:

\[
80\%
\]

---

# 24. AUC-PR / Average Precision

Because of LANL's extreme class imbalance, Precision-Recall analysis is particularly useful.

The Precision-Recall curve studies the relationship between:

\[
Precision
\]

and:

\[
Recall
\]

across different anomaly thresholds.

Average Precision (AP) summarizes performance across the Precision-Recall curve.

This avoids selecting only one arbitrary decision threshold.

For the thesis, appropriate metrics therefore include:

```text
Precision
Recall
False Positive Rate (FPR)
AUC-PR / Average Precision (AP)
```

ROC-AUC can also be reported, but Precision-Recall-based metrics are particularly informative under severe class imbalance.

---

# 25. Why Preprocessing Matters So Much

The paper re-evaluates several existing graph-based lateral movement detectors using its standardized preprocessing methodology.

Reported performance decreases substantially.

For example, the paper reports the following Average Precision values for LANL:

| Method | New AP | Previously Reported AP |
|---|---:|---:|
| Euler | approximately 0.02 | 5.23 |
| Argus | approximately 0.09 | 32.27 |

This demonstrates an important principle:

\[
\boxed{
\text{Preprocessing decisions}
\rightarrow
\text{Large changes in measured performance}
}
\]

Therefore, high reported performance from previous research should not automatically be considered directly comparable unless the studies use equivalent:

```text
datasets
splits
filters
labels
graph construction
evaluation units
metrics
```

Table 1. Preprocessing choices for the LANL dataset. Source: [Larroche (2026) PDF](https://www.google.com/search?q=On+Fair+and+Realistic+Performance+Evaluations+for+Graph-Based+Lateral+Movement+Detectors+pdf).

| Detector | Train/test split | Filtering steps | Merging |
|---|---|---|---|
| GL-GV [6] | Train: 40 days without red team activity. Test: 18 remaining days. | Computer accounts removed from test set. | Train: one edge per source/dest. pair. Test: no merging. |
| MLTracer [19] | Train: first 12 days. Test: next 18 days. | Only compromised users. | No merging. |
| Pikachu [23] | Train: first 40 hours. Test: rest of the dataset. | Computer and built-in accounts removed. Subsampling of users. Remote LogOn events only. | No merging. |
| Euler [17] | Train: first 41 hours. Test: rest of the dataset. | Remote NTLM LogOn events only. | One edge per snapshot and source/dest. pair. |
| Argus [30] | Train: first 41 hours. Test: rest of first 14 days. | Remote NTLM LogOn events only. | One edge per snapshot and source/dest. pair. |
| Jbeil [14] | Train: first 85% of the dataset for a subset of nodes. Test: last 15% of the dataset. | NTLM LogOn events only. | No merging. |

---

# 26. Fair DeepCASE vs GRU+GNN Comparison

The proposed experiment should ensure that both architectures receive exactly the same underlying information.

Conceptually:

```text
                   LANL
                     │
                     ▼
               Preprocessing
                     │
                     ▼
              Temporal Split
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼

         DeepCASE       DeepCASE + GNN
             │                │
             ▼                ▼

        predictions       predictions
             │                │
             └───────┬────────┘
                     │
                     ▼

            Same test events
                     │
                     ▼

            Same Red Team labels
                     │
                     ▼

       Same evaluation metrics
```

This helps isolate the experimental variable:

\[
\text{Cross-host graph modeling}
\]

rather than accidentally measuring preprocessing differences.

---

# 27. Recommended LANL Preprocessing Pipeline

Based on the paper, a possible preprocessing pipeline is:

```text
Raw LANL auth.txt
        │
        ▼
Select remote LogOn events
        │
        ▼
Keep authentication packages
        │
        ├── NTLM
        ├── Kerberos
        └── Negotiate
        │
        ▼
Keep relevant account categories
        │
        ├── normal users
        ├── computer accounts
        ├── local accounts
        └── built-in accounts
        │
        ▼
Do NOT downsample benign users
        │
        ▼
Strict temporal split
        │
        ├── TRAIN
        │     first 41 hours
        │
        └── TEST
              remaining dataset
        │
        ▼
Build temporal graph snapshots
        │
        ▼
G_t = (V_t,E_t,X_t,A_t)
        │
        ▼
GRU + GraphSAGE
        │
        ▼
Anomaly score
        │
        ▼
Map graph predictions to events
        │
        ▼
Original LANL Red Team labels
        │
        ▼
Evaluation
```

---

# 28. Proposed Graph Representation

For temporal window \(t\):

\[
G_t=(V_t,E_t,X_t,A_t)
\]

where:

### Nodes

\[
V_t=\{\text{active computers during window }t\}
\]

For example:

```text
C10
C20
C30
C40
```

### Edges

An edge:

\[
(C_i,C_j)\in E_t
\]

exists when an authentication event occurs from:

\[
C_i\rightarrow C_j
\]

during temporal window \(t\).

### Node Features

Potential node features include:

```text
GRU contextual embedding
authentication frequency
unique users
number of destination hosts
number of source hosts
failed authentication count
successful authentication count
```

### Edge Features

Potential edge features include:

```text
authentication type
logon type
success/failure
user/account type
event frequency
timestamp information
```

The exact feature set remains an experimental design decision and is not prescribed by the paper.

---

# 29. Proposed GRU + GraphSAGE Flow

A conceptual architecture could be:

```text
LANL Authentication Events
            │
            ▼
     Temporal Sequences
            │
            ▼
           GRU
            │
            ▼
    Context Embeddings
            │
            ▼
       Host Graph G_t
            │
            ▼
        GraphSAGE
            │
            ▼
 Cross-Host Representation
            │
            ▼
      Anomaly Detection
            │
            ▼
       Anomaly Score
```

The GRU captures:

```text
Temporal context
```

while GraphSAGE captures:

```text
Cross-host context
```

Conceptually:

\[
\boxed{
\text{Temporal Context}
+
\text{Cross-Host Context}
\rightarrow
\text{Enhanced Event Representation}
}
\]

---

# 30. Important Experimental Principle

The central experimental principle derived from this paper is:

\[
\boxed{
\text{Same Data}
+
\text{Same Preprocessing}
+
\text{Same Labels}
+
\text{Same Evaluation}
}
\]

for:

\[
DeepCASE
\]

and:

\[
DeepCASE + GRU/GNN
\]

The architecture should be the experimental variable.

Preprocessing should **not** become an uncontrolled experimental variable.

---

# 31. Recommended Experimental Structure

A strong experimental design could therefore be:

## Baseline

```text
DeepCASE
```

using the selected LANL authentication events.

## Proposed Model

```text
DeepCASE + Cross-Host GraphSAGE
```

using the exact same LANL events.

## Controlled Variables

Keep constant:

```text
LANL data
train/test split
Red Team labels
event filters
evaluation events
random seeds when applicable
metrics
```

## Independent Variable

Change:

```text
Cross-host graph modeling
```

## Dependent Variables

Measure:

\[
Precision
\]

\[
Recall
\]

\[
FPR
\]

\[
AUC\text{-}PR
\]

\[
AP
\]

---

# 32. Potential Snapshot Experiment

Because snapshot size can affect graph structure, an additional experiment could investigate:

\[
\Delta\in
\{
5\text{ min},
15\text{ min},
30\text{ min},
60\text{ min}
\}
\]

For each value of \(\Delta\):

```text
LANL events
     │
     ▼
Graph snapshots
     │
     ▼
GRU + GraphSAGE
     │
     ▼
Same event-level evaluation
```

This would allow analysis of the effect of temporal aggregation without changing the underlying test events.

However, snapshot-size tuning should be performed using validation data rather than selecting the value that performs best on the final test set.

---

# 33. Threats to Validity

Based on this paper, several threats to validity should be explicitly acknowledged.

## LANL Contains One Main Simulated Intrusion

The malicious authentication events correspond to one simulated Red Team intrusion.

Therefore, results may not generalize to all forms of lateral movement.

## Most Malicious Events Originate from C17693

This creates a structural bias in the malicious portion of the dataset.

A model could potentially learn characteristics associated with this particular host or attack trajectory.

## Labels Are Incomplete

LANL labels represent only a subset of Red Team-related events.

Therefore:

```text
unlabeled ≠ guaranteed benign
```

in every possible interpretation.

## Snapshot Size Changes Graph Structure

Different values of:

\[
\Delta
\]

produce different graph structures.

This can affect model behavior and evaluation.

## Extreme Class Imbalance

Malicious events are extremely rare compared with benign authentication events.

This makes accuracy unsuitable and makes false-positive behavior particularly important.

---

# 34. Main Takeaways for the Thesis

The most important lessons from this paper are:

1. **Use strict temporal separation between training and testing.**

2. **Consider training on the first 41 hours of LANL and testing on the remainder.**

3. **Do not restrict authentication events to NTLM simply because LANL Red Team events use NTLM.**

4. **Do not automatically remove computer, local, or built-in accounts.**

5. **Do not downsample benign users based on knowledge of which accounts were compromised.**

6. **Use the original LANL Red Team labels rather than creating additional attack labels without strong justification.**

7. **Treat snapshot size as an important graph-construction parameter.**

8. **Do not allow graph aggregation to unfairly change the evaluation population.**

9. **Prefer event-level evaluation when comparing architectures.**

10. **Use Precision, Recall, FPR, and Precision-Recall-based metrics such as AUC-PR/AP because of the extreme class imbalance.**

11. **Do not train a supervised model using portions of the same LANL Red Team intrusion that will later be used for testing.**

12. **Explicitly acknowledge that LANL contains a limited Red Team scenario and that almost all labeled malicious authentications originate from C17693.**

13. **Use identical preprocessing and evaluation procedures for DeepCASE and the proposed GRU + GNN architecture.**

---

# 35. Practical Rule for Implementation

A useful rule while implementing the LANL preprocessing pipeline is:

> **Do not remove an event simply because it makes the detection problem harder. Remove it only when there is a defensible, model-independent reason for excluding it.**

And for evaluation:

> **Graph construction may aggregate events for representation and message passing, but the evaluation should preserve the original event-level ground truth whenever possible.**

---

# 36. Relationship With the Thesis Hypothesis

The thesis investigates whether incorporating cross-host relationships improves contextual anomaly detection beyond DeepCASE's host-local context.

The experiment can be conceptualized as:

\[
H_0:
Performance_{DeepCASE}
=
Performance_{GRU+GNN}
\]

versus:

\[
H_1:
Performance_{GRU+GNN}
>
Performance_{DeepCASE}
\]

for selected detection metrics while controlling false positives.

However, a valid test of this hypothesis requires ensuring that:

\[
Dataset_{baseline}
=
Dataset_{proposed}
\]

\[
Labels_{baseline}
=
Labels_{proposed}
\]

\[
Evaluation_{baseline}
=
Evaluation_{proposed}
\]

so that the main meaningful difference is:

\[
\boxed{\text{Cross-host graph context}}
\]

This paper provides important methodological guidance for achieving that controlled comparison.

---

# Reference

Larroche, C. (2026). *On Fair and Realistic Performance Evaluations for Graph-Based Lateral Movement Detectors*. French National Cybersecurity Agency (ANSSI).

The paper focuses on the LANL and DARPA OpTC datasets and proposes standardized preprocessing and labeling practices for more realistic and fair evaluation of graph-based lateral movement detectors.
