# LANL Cyber Security Dataset — Structure and Code Reference

## 1. Overview

The **Los Alamos National Laboratory (LANL) Comprehensive, Multi-Source Cyber-Security Events Dataset** contains approximately **58 days of enterprise network activity**.

The dataset includes several types of security events:

* Authentication events
* Process activity
* DNS queries
* Network flows
* Red Team attack events

The dataset contains temporal and relational information, making it particularly useful for studying **cross-host behavior**, **lateral movement**, **sequence modeling**, and **Graph Neural Networks (GNNs)**.

Official dataset documentation:

`https://csr.lanl.gov/data/cyber1/`

---

# 2. Understanding LANL Identifiers

One of the most important characteristics of the dataset is that much of the information has been **anonymized**.

For example:

```text
C553
U147
P16
N10471
DOM1
```

These identifiers generally do **not** correspond to their original real-world names.

Instead, LANL replaced sensitive information with consistent anonymous identifiers.

## Common identifiers

| Code     | Meaning                  | Example       |
| -------- | ------------------------ | ------------- |
| `U###`   | Anonymized user          | `U748`        |
| `C###`   | Anonymized computer/host | `C553`        |
| `C###$`  | Computer/machine account | `C553$`       |
| `P###`   | Anonymized process       | `P16`         |
| `N###`   | Anonymized network port  | `N10471`      |
| `DOM#`   | Anonymized domain        | `DOM1`        |
| `?`      | Missing/unknown value    | `?`           |
| `SYSTEM` | Windows SYSTEM account   | `SYSTEM@C653` |

For example:

```text
C553
```

should be interpreted approximately as:

> Computer/host number 553 in the anonymized dataset.

It does **not** mean that the computer is a particular model or type.

Similarly:

```text
U147
```

means:

> Anonymized user 147.

The important property is that identifiers are consistent throughout the dataset.

Therefore, if:

```text
C553
```

appears in authentication, process, DNS, or network-flow data, it represents the same anonymized computer.

This consistency is extremely useful when constructing graphs.

---

# 3. Understanding the Timestamp

LANL does not provide the original calendar timestamps.

Instead, time is represented as the number of **seconds since the beginning of data collection**.

The dataset starts approximately at:

```text
t = 1
```

Therefore:

```text
1       → beginning of dataset
60      → approximately 1 minute
3600    → approximately 1 hour
86400   → approximately 1 day
172800  → approximately 2 days
```

The timestamp should therefore be interpreted as a relative timestamp rather than a Unix timestamp.

## Converting seconds to hours

Given timestamp:

```text
151648
```

the approximate number of hours is:

$$
\frac{151648}{3600} \approx 42.12
$$

Therefore, the event occurred approximately **42.1 hours after the beginning of data collection**.

## Converting seconds to days

$$
\frac{151648}{86400} \approx 1.755
$$

Therefore, the event occurred approximately **1.76 days after the beginning of the dataset**.

The actual calendar date is not required for most machine-learning experiments because relative temporal relationships are preserved.

For example:

$$
\Delta t = t_j - t_i
$$

can measure the amount of time between two security events.

This is particularly useful for temporal windows and sequence modeling.

---

# 4. Authentication Data — `auth.txt.gz`

Authentication events are particularly important for cross-host modeling and lateral-movement detection.

The structure is approximately:

```text
time,
source user@domain,
destination user@domain,
source computer,
destination computer,
authentication type,
logon type,
authentication orientation,
success/failure
```

Example:

```text
1,C625$@DOM1,U147@DOM1,C625,C625,Negotiate,Batch,LogOn,Success
```

## Interpretation

| Field                | Value        | Meaning                          |
| -------------------- | ------------ | -------------------------------- |
| Time                 | `1`          | Second 1 of dataset              |
| Source user          | `C625$@DOM1` | Computer account C625            |
| Destination user     | `U147@DOM1`  | User 147                         |
| Source computer      | `C625`       | Computer 625                     |
| Destination computer | `C625`       | Computer 625                     |
| Authentication type  | `Negotiate`  | Windows Negotiate authentication |
| Logon type           | `Batch`      | Batch logon                      |
| Orientation          | `LogOn`      | Logon event                      |
| Result               | `Success`    | Authentication succeeded         |

---

# 5. Understanding `$` in Usernames

A value such as:

```text
C625$
```

represents a **computer/machine account** rather than an ordinary human user account.

For example:

```text
U147
```

represents an anonymized user.

Meanwhile:

```text
C625$
```

represents a computer account.

The complete value:

```text
C625$@DOM1
```

can be interpreted as:

```text
C625$@DOM1
│     │
│     └── Domain 1
│
└──────── Machine/computer account C625
```

This distinction can be important when analyzing authentication behavior.

---

# 6. Authentication Types

Authentication records can contain values such as:

```text
Kerberos
NTLM
Negotiate
```

### Kerberos

A common authentication protocol used in Windows enterprise environments.

### NTLM

An older Windows authentication mechanism.

### Negotiate

Windows determines which available authentication mechanism should be used.

For example, Windows may negotiate between Kerberos and NTLM.

---

# 7. Logon Orientation

Authentication events can include values such as:

```text
LogOn
LogOff
```

These represent the orientation or type of authentication event.

For example:

```text
LogOn
```

represents the creation of a logon session.

---

# 8. Authentication Result

Authentication events can also indicate whether authentication succeeded.

Examples:

```text
Success
Fail
```

This information can potentially be useful as an edge feature in a graph model.

For example:

```text
C100 ──Success──→ C200
```

versus:

```text
C100 ──Fail──→ C200
```

---

# 9. Process Data — `proc.txt.gz`

Process activity has approximately the following structure:

```text
time,user@domain,computer,process name,start/end
```

Example:

```text
1,C553$@DOM1,C553,P16,Start
1,C553$@DOM1,C553,P25,End
1,C553$@DOM1,C553,P25,Start
```

Consider:

```text
1,C553$@DOM1,C553,P16,Start
```

This means approximately:

> At second 1, machine account `C553$` on domain `DOM1`, operating on computer `C553`, started anonymized process `P16`.

The identifiers can be decomposed as:

```text
C553$@DOM1
│    │    │
│    │    └── Domain
│    │
│    └─────── $ indicates machine account
│
└──────────── Computer/account identifier
```

---

# 10. What Does `P16` Mean?

A value such as:

```text
P16
```

is an anonymized process identifier.

It should **not** automatically be interpreted as Windows Process ID 16.

The original process may have represented something such as:

```text
powershell.exe
cmd.exe
svchost.exe
chrome.exe
```

but the original process name has been anonymized.

Therefore:

```text
P16
```

should generally be treated as a categorical identifier:

```text
process = P16
```

rather than attempting to determine the original executable.

---

# 11. DNS Data — `dns.txt.gz`

DNS data has approximately the following structure:

```text
time,source computer,computer resolved
```

Example:

```text
31,C161,C2109
35,C5642,C528
38,C3380,C22841
```

Consider:

```text
31,C161,C2109
```

This means approximately:

> At second 31, computer `C161` made a DNS lookup that resolved to computer `C2109`.

Graphically:

```text
C161 ───── DNS lookup ─────→ C2109
```

This provides another possible relationship between computers.

---

# 12. Network Flow Data — `flows.txt.gz`

Network flow data has approximately the following structure:

```text
time,
duration,
source computer,
source port,
destination computer,
destination port,
protocol,
packet count,
byte count
```

Example:

```text
1,9,C3090,N10471,C3420,N46,6,3,144
```

## Interpretation

| Field                |    Value | Meaning                     |
| -------------------- | -------: | --------------------------- |
| Time                 |      `1` | Beginning of dataset        |
| Duration             |      `9` | Flow lasted 9 seconds       |
| Source computer      |  `C3090` | Source host                 |
| Source port          | `N10471` | Anonymized source port      |
| Destination computer |  `C3420` | Destination host            |
| Destination port     |    `N46` | Anonymized destination port |
| Protocol             |      `6` | IP protocol number          |
| Packets              |      `3` | 3 packets                   |
| Bytes                |    `144` | 144 bytes                   |

Graphically:

```text
             TCP / Protocol 6
             3 packets
             144 bytes

C3090 ───────────────────────→ C3420
```

---

# 13. Understanding `N10471`

Values such as:

```text
N10471
N46
```

represent anonymized network-port identifiers.

They should generally be treated as categorical values.

Some well-known ports may remain recognizable, while other port information has been anonymized.

Therefore, an identifier such as:

```text
N10471
```

does not necessarily mean TCP/UDP port `10471`.

It represents an anonymized port identifier.

---

# 14. Understanding Protocol `6`

Unlike anonymized identifiers such as `N10471`, the protocol field can contain standard IP protocol numbers.

For example:

```text
6  → TCP
17 → UDP
```

Therefore:

```text
1,9,C3090,N10471,C3420,N46,6,3,144
```

represents TCP network traffic between two anonymized computers.

---

# 15. Red Team Data — `redteam.txt.gz`

The Red Team file is particularly important because it identifies known malicious authentication activity.

Its structure is approximately:

```text
time,user@domain,source computer,destination computer
```

Example:

```text
151648,U748@DOM1,C17693,C728
151993,U6115@DOM1,C17693,C1173
153792,U636@DOM1,C17693,C294
```

Consider:

```text
151648,U748@DOM1,C17693,C728
```

This represents a known Red Team event involving:

```text
Time             = 151648
User             = U748
Domain           = DOM1
Source computer  = C17693
Destination      = C728
```

Graphically:

```text
                 U748
                  │
                  │ credentials
                  ▼

C17693 ───────────────────→ C728
 Source                      Destination
             RED TEAM
```

These events provide **ground truth** for known malicious behavior.

---

# 16. Relationship Between Authentication and Red Team Data

Suppose `auth.txt` contains:

```text
151648,U748@DOM1,U748@DOM1,C17693,C728,Kerberos,Network,LogOn,Success
```

and `redteam.txt` contains:

```text
151648,U748@DOM1,C17693,C728
```

The combination:

```text
timestamp = 151648
user      = U748
source    = C17693
dest      = C728
```

can be used to determine that the authentication event corresponds to known Red Team activity.

Conceptually:

```text
auth.txt

C17693 ─────→ C728
      authentication
            │
            │ match
            ▼

redteam.txt

C17693 ─────→ C728
       RED TEAM
```

This allows authentication events to be labeled for evaluation.

---

# 17. Why LANL Is Useful for Cross-Host Graph Modeling

Consider the following simplified authentication events:

```text
100,U1@DOM1,U1@DOM1,C10,C20,Kerberos,Network,LogOn,Success
105,U1@DOM1,U1@DOM1,C20,C30,Kerberos,Network,LogOn,Success
112,U1@DOM1,U1@DOM1,C30,C40,Kerberos,Network,LogOn,Success
```

These events can be converted into:

```text
C10 ─────→ C20 ─────→ C30 ─────→ C40
 t=100      t=105       t=112
```

The computers become graph nodes:

$$
V = \{C10,C20,C30,C40\}
$$

The authentication relationships become edges:

$$
E =
\{
(C10,C20),
(C20,C30),
(C30,C40)
\}
$$

The timestamps preserve the temporal progression:

$$
C10
\xrightarrow{t=100}
C20
\xrightarrow{t=105}
C30
\xrightarrow{t=112}
C40
$$

This structure is highly relevant for studying **lateral movement**.

---

# 18. Temporal Graph Representation

Instead of constructing one graph containing the entire 58-day dataset, the events can be divided into temporal windows.

For example:

```text
Window 1
0–300 seconds

Window 2
300–600 seconds

Window 3
600–900 seconds
```

Each window produces a graph:

$$
G_t = (V_t,E_t,X_t,A_t)
$$

where:

* \(V_t\): active hosts during temporal window \(t\)
* \(E_t\): relationships between hosts
* \(X_t\): node features
* \(A_t\): edge attributes

For authentication events, a directed edge can be constructed as:

$$
C_i \rightarrow C_j
$$

when computer \(C_i\) authenticates against computer \(C_j\).

---

# 19. Possible Graph Edge Features

An authentication relationship can contain more information than simply:

```text
C10 → C20
```

An edge could contain:

```text
Source host
Destination host
Timestamp
User
Authentication type
Logon type
Success/failure
```

Mathematically:

$$
e_{ij}
=
(
C_i,
C_j,
t,
u,
a,
l,
r
)
$$

where:

* \(C_i\) = source computer
* \(C_j\) = destination computer
* \(t\) = timestamp
* \(u\) = user
* \(a\) = authentication type
* \(l\) = logon type
* \(r\) = authentication result

For example:

```text
C10 ─────────────────────────────→ C20
     Kerberos
     Network
     Success
     User U1
     t = 100
```

---

# 20. Multiple Relationship Types

LANL contains several possible relationships between computers.

For example:

### Authentication

```text
C100 ── authentication ──→ C200
```

### DNS

```text
C100 ── DNS lookup ──→ C200
```

### Network Flow

```text
C100 ── network traffic ──→ C200
```

Potentially, this could be represented as a multi-relational graph:

$$
e_{ij} = (C_i,C_j,\text{event type},t)
$$

For example:

```text
             Authentication
        ┌────────────────────→ C20
        │
        │    DNS
C10 ────┼────────────────────→ C30
        │
        │    Network Flow
        └────────────────────→ C40
```

However, starting with **authentication events only** can provide a simpler and more controlled experimental design.

---

# 21. Relevance to GRU + GNN Modeling

The LANL dataset contains two particularly useful structures.

## Temporal Structure

Events occur sequentially:

```text
e1 → e2 → e3 → e4 → ...
```

A sequential model such as a **GRU** can model these temporal dependencies.

For example:

$$
h_t = GRU(x_t,h_{t-1})
$$

where:

* \(x_t\) = security event at time \(t\)
* \(h_{t-1}\) = previous context
* \(h_t\) = current contextual representation

## Relational Structure

Computers interact with other computers:

```text
C10 ──→ C20
          │
          ▼
        C30
          │
          ▼
        C40
```

A **Graph Neural Network** can propagate information between these connected computers.

For example:

$$
h_v^{(k+1)}
=
\sigma
\left(
W
\cdot
AGGREGATE
\left(
\{h_u^{(k)} : u \in N(v)\}
\right)
\right)
$$

This allows the model to incorporate information from neighboring hosts.

---

# 22. Conceptual GRU + GNN Architecture

A simplified architecture can therefore be viewed as:

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
      GraphSAGE/GNN
           │
           ▼
 Cross-Host Representation
           │
           ▼
    Anomaly Detection
```

The GRU answers:

> What has been happening over time?

The GNN answers:

> What is happening between connected computers?

Together:

```text
             TEMPORAL
                │
                ▼
Events ───────► GRU
                │
                ▼
            Embeddings
                │
                ▼
Hosts ────────► GNN ◄──── Host relationships
                │
                ▼
          Cross-host context
```

---

# 23. Example of Lateral Movement

Suppose LANL contains:

```text
100,U25@DOM1,U25@DOM1,C10,C20,Kerberos,Network,LogOn,Success
115,U25@DOM1,U25@DOM1,C20,C35,Kerberos,Network,LogOn,Success
127,U25@DOM1,U25@DOM1,C35,C70,Kerberos,Network,LogOn,Success
```

This could indicate the following progression:

```text
U25
 │
 ▼

C10
 │
 │ t=100
 ▼
C20
 │
 │ t=115
 ▼
C35
 │
 │ t=127
 ▼
C70
```

A host-local sequence model may analyze the events associated with each computer individually.

A graph model can explicitly represent:

$$
C10 \rightarrow C20 \rightarrow C35 \rightarrow C70
$$

This is one of the motivations for using cross-host graph modeling for lateral-movement detection.

---

# 24. Important Point About Anonymization

It is generally unnecessary to recover the real-world meaning of:

```text
C553
U147
P16
N10471
```

For machine learning, these identifiers can still provide useful structure.

For example:

```text
C553 → graph node
U147 → categorical user
P16  → categorical process
```

The important information is the **relationship between entities**.

For example:

```text
U147
  │
  ▼
C100 ─────→ C553 ─────→ C900
```

The model does not need to know the real hostname of `C553` to learn that it acts as an intermediate host connecting `C100` and `C900`.

---

# 25. Quick Reference

```text
LANL IDENTIFIERS
================

U###      = anonymized user
C###      = anonymized computer/host
C###$     = computer/machine account
P###      = anonymized process
N###      = anonymized network port
DOM#      = anonymized domain
?         = missing/unknown value

TIME
====

1         ≈ beginning of dataset
60        ≈ 1 minute
3600      ≈ 1 hour
86400     ≈ 1 day

AUTHENTICATION
==============

Kerberos  = Kerberos authentication
NTLM      = NTLM authentication
Negotiate = Windows authentication negotiation

RESULT
======

Success   = successful authentication
Fail      = failed authentication

ORIENTATION
===========

LogOn     = logon event
LogOff    = logoff event

NETWORK PROTOCOL
================

6         = TCP
17        = UDP
```

---

# 26. Main Files

| File             | Information             | Potential Thesis Use              |
| ---------------- | ----------------------- | --------------------------------- |
| `auth.txt.gz`    | Authentication activity | Main GRU/GNN input                |
| `proc.txt.gz`    | Process activity        | Additional host behavior          |
| `dns.txt.gz`     | DNS queries             | Additional host relationships     |
| `flows.txt.gz`   | Network flows           | Network-level graph relationships |
| `redteam.txt.gz` | Known Red Team events   | Ground truth/evaluation           |

For an initial cross-host implementation, the most important files are:

```text
auth.txt.gz
     +
redteam.txt.gz
```

because `auth.txt.gz` provides the temporal host-to-host authentication relationships and `redteam.txt.gz` provides known malicious Red Team events.

---

# 27. Main Idea for the Thesis

The LANL dataset can be viewed as a temporal enterprise graph.

Instead of seeing:

```text
151648,U748@DOM1,C17693,C728
```

as simply a CSV row, it can be interpreted as a relationship:

```text
                U748
                  │
                  ▼

C17693 ─────────────────→ C728
 source                    destination
        t = 151648
```

Multiple events create a temporal graph:

```text
               t1              t2
C10 ───────────────→ C20 ───────────────→ C30
                       │
                       │ t3
                       ▼
                      C40
```

This can be represented as:

$$
G_t=(V_t,E_t,X_t,A_t)
$$

and analyzed using a Graph Neural Network such as **GraphSAGE**.

The fundamental idea is therefore:

$$
\boxed{
\text{Temporal context}
+
\text{Cross-host relationships}
\rightarrow
\text{Improved contextual representation}
}
$$

where temporal context can be learned by the **GRU** and cross-host relationships can be modeled by the **GNN**.

---

# References

Los Alamos National Laboratory. *Comprehensive, Multi-Source Cyber-Security Events*. LANL Cyber Security Dataset.

Official documentation:

`https://csr.lanl.gov/data/cyber1/`

Kent, A. D. (2015). *Comprehensive, Multi-Source Cyber-Security Events*. Los Alamos National Laboratory.

