# Paper: [text](<../../../../../../Downloads/Adaptive temporal graph neural networks for detecting coordinated multi-stage cyber-attacks in enterprise systems.pdf>)

- In GNNs, information from neighboring nodes is aggregated through *message passing*. My idea is to use a weight to identify threats.


# Types of GNN

## GraphSAGE
We will use *GraphSAGE* ==> "Ask my neighbors what they are doing." If there are thousands of neighbors, this can be expensive, so GraphSAGE takes a sample of them.

## GCN "Average information from everyone around me"


## GAT -- "Some neighbors matter more than others" --->> CONSIDER


## GIN (Graph Isomorphism Network) -- "I want to distinguish graph structures very carefully"


Note: The paper says, "GNN-based models are also criticised for overemphasising predictive performance and, thus, underemphasising explainability, which is valuable in operational security contexts, as analysts are expected to review the alerts and respond."

So it only says whether something is malicious. The context is important: Who is the user? What is the attack? And so on.


# Main goal
"Instead of looking only at a fixed network graph, our model can analyze how the network changes over time and learn which moments in that evolution are particularly important for detecting a coordinated attack."

GNN + Dynamic Graphs + Temporal attention
