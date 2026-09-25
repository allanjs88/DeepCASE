"""High-level fit, score, calibrate, predict, save and load interface."""

import random

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from .context_builder import ContextBuilder
from .interpreter import Interpreter
from .preprocessing import Preprocessor


class DeepCASEGNN:
    """First-version self-supervised GRU + GraphSAGE event detector.

    Learns P(event category | earlier host histories, graph, ordered endpoints).
    Higher negative log likelihood means more surprising, not necessarily
    malicious. Red-team labels never enter training, graphs, or calibration.
    """

    def __init__(self, max_length=10, window_seconds=300, hidden_size=64,
                 embedding_size=32, gnn_layers=2, dropout=0.1, seed=0, device="cpu"):
        self.config = dict(max_length=max_length, window_seconds=window_seconds,
                           hidden_size=hidden_size, embedding_size=embedding_size,
                           gnn_layers=gnn_layers, dropout=dropout, seed=seed)
        self.device = torch.device(device)
        self.preprocessor = Preprocessor(max_length, window_seconds)
        self.context_builder = None
        self.interpreter = Interpreter()
        self.loss_history = []

    def _initialize(self):
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.context_builder = ContextBuilder(
            features=self.preprocessor.features,
            **{k: self.config[k] for k in ("hidden_size", "embedding_size", "gnn_layers", "dropout")}
        ).to(self.device)

    def fit(self, frame, epochs=10, batch_size=8, learning_rate=0.001, verbose=False):
        """Train afresh using only the supplied training partition.

        Batch size counts timestamp snapshots, not events. Every event at a
        timestamp shares the same strictly earlier graph. Loss is event weighted.
        """
        if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive.")
        dataset = self.preprocessor.fit_transform(frame)
        self._initialize()
        self.interpreter = Interpreter()
        self.loss_history = []
        optimizer = torch.optim.Adam(self.context_builder.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.config["seed"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        for epoch in range(epochs):
            self.context_builder.train()
            total_loss = total_events = 0
            for graph in loader:
                graph = graph.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.context_builder(graph), graph.y - 1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.context_builder.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * graph.y.numel()
                total_events += graph.y.numel()
            self.loss_history.append(total_loss / total_events)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: event NLL={self.loss_history[-1]:.6f}")
        return self

    def score(self, frame, history=None, batch_size=8):
        """Return chronological event IDs and scores, with optional labels intact.

        Earlier rows of `frame` become observed history for later rows. Pass the
        previous partition explicitly as `history`; this never updates weights.
        Returned rows are chronological, not necessarily input order: join by ID.
        """
        if self.context_builder is None:
            raise RuntimeError("Fit or load a model before scoring.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        dataset = self.preprocessor.transform(frame, history=history)
        scores = np.empty(len(dataset.frame), dtype=float)
        self.context_builder.eval()
        with torch.no_grad():
            for graph in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                graph = graph.to(self.device)
                loss = F.cross_entropy(self.context_builder(graph), graph.y - 1, reduction="none")
                scores[graph.row_position.cpu().numpy()] = loss.cpu().numpy()
        columns = ["event_id", "timestamp"] + (["label"] if "label" in dataset.frame else [])
        result = dataset.frame[columns].copy()
        result["score"] = scores
        return result

    def calibrate(self, validation, history=None, quantile=0.99, batch_size=8):
        """Freeze a label-free validation surprise quantile as the alert cutoff."""
        scores = self.score(validation, history=history, batch_size=batch_size)
        return self.interpreter.calibrate(scores.score.to_numpy(), quantile)

    def predict(self, frame, history=None, batch_size=8):
        result = self.score(frame, history=history, batch_size=batch_size)
        result["prediction"] = self.interpreter.predict(result.score.to_numpy())
        return result

    def save(self, path):
        """Store weights, training vocabulary, configuration and threshold.

        Raw data/history is intentionally supplied separately at inference.
        """
        if self.context_builder is None:
            raise RuntimeError("There is no fitted model to save.")
        torch.save({"format_version": 1, "config": self.config,
                    "vocabulary": self.preprocessor.vocabulary,
                    "state_dict": self.context_builder.state_dict(),
                    "threshold": self.interpreter.threshold,
                    "loss_history": self.loss_history}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported checkpoint version.")
        model = cls(**checkpoint["config"], device=device)
        model.preprocessor.vocabulary = checkpoint["vocabulary"]
        model._initialize()
        model.context_builder.load_state_dict(checkpoint["state_dict"])
        model.context_builder.eval()
        model.interpreter.threshold = checkpoint["threshold"]
        model.loss_history = checkpoint["loss_history"]
        return model
