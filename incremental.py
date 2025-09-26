"""Incremental learning utilities for scFASTopic.

This module draws inspiration from StreamETM, which leverages unbalanced
optimal transport (OT) to align topic distributions across streaming tasks
and mitigate catastrophic forgetting.  We adapt the idea to FASTopic by
aligning the topic embeddings learned on new batches with a reference set of
previous topics using unbalanced OT from the POT library.

Key components
--------------
- ``UnbalancedOTAligner`` computes a soft assignment between previous and new
  topic embeddings via Sinkhorn-based unbalanced OT.
- ``IncrementalTrainer`` wraps a FASTopic instance, orchestrating fine-tuning
  on a new batch followed by OT-based topic alignment and optional historical
  state updates.

The implementation assumes that each incremental batch shares the same gene
vocabulary as the model (or is pre-aligned beforehand) and that the model
uses the ``fit_transform_sc`` interface for single-cell data.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import ot
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "The POT package is required for incremental learning utilities. "
        "Install it via `pip install pot`."
    ) from exc

from fastopic.FASTopic import FASTopic


@dataclass
class AlignmentResult:
    """Container returned by :class:`UnbalancedOTAligner`.

    Attributes
    ----------
    coupling: np.ndarray
        Transport plan (previous_topics x new_topics).
    prev_weights: np.ndarray
        Normalised weights for the previous topics.
    new_weights: np.ndarray
        Normalised weights for the new topics.
    transported_mass_prev: np.ndarray
        Mass sent from each previous topic (size ``prev_topics``).
    transported_mass_new: np.ndarray
        Mass received by each new topic (size ``new_topics``).
    barycentric_projection: np.ndarray
        Barycentric projection of new topics expressed in the space of
        previous topics.  This can be used to initialise or regularise the
        updated topic embeddings.
    unmatched_mask: np.ndarray
        Boolean mask identifying new topics that received less mass than the
        configured ``min_transport_mass`` threshold during alignment.
    """

    coupling: np.ndarray
    prev_weights: np.ndarray
    new_weights: np.ndarray
    transported_mass_prev: np.ndarray
    transported_mass_new: np.ndarray
    barycentric_projection: np.ndarray
    unmatched_mask: np.ndarray


class UnbalancedOTAligner:
    """Align topic embeddings across incremental updates.

    Parameters
    ----------
    reg : float
        Entropic regularisation strength for the Sinkhorn solver.
    reg_m : float
        Mass regularisation strength controlling how much mass can be created
        or destroyed (``reg_m`` in POT's terminology).  Lower values encourage
        mass conservation, higher values allow more flexibility.
    metric : str
        Distance metric for the cost matrix (passed to ``ot.dist``).
    min_transport_mass : float
        Threshold on the transported mass used to flag topics that were not
        matched strongly to any previous topic.
    smoothing : float
        Blend factor between the raw new topics and the barycentric projection
        (``0`` keeps new topics, ``1`` fully adopts the projection).
    """

    def __init__(
        self,
        *,
        reg: float = 0.05,
        reg_m: float = 10.0,
        metric: str = "euclidean",
        min_transport_mass: float = 1e-3,
        smoothing: float = 0.5,
    ) -> None:
        self.reg = reg
        self.reg_m = reg_m
        self.metric = metric
        self.min_transport_mass = min_transport_mass
        self.smoothing = smoothing

    def align(
        self,
        prev_topics: np.ndarray,
        new_topics: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
        new_weights: Optional[np.ndarray] = None,
    ) -> AlignmentResult:
        if prev_topics.ndim != 2 or new_topics.ndim != 2:
            raise ValueError("Topic embeddings must be 2-D arrays")
        if prev_topics.shape[1] != new_topics.shape[1]:
            raise ValueError("Embedding dimensionality mismatch between batches")

        n_prev, n_new = prev_topics.shape[0], new_topics.shape[0]

        prev_weights = self._normalise_weights(prev_weights, n_prev)
        new_weights = self._normalise_weights(new_weights, n_new)

        cost = ot.dist(prev_topics.astype(np.float64), new_topics.astype(np.float64), metric=self.metric)
        coupling = ot.unbalanced.sinkhorn_unbalanced(
            prev_weights,
            new_weights,
            cost,
            reg=self.reg,
            reg_m=self.reg_m,
        )

        transported_mass_prev = coupling.sum(axis=1)
        transported_mass_new = coupling.sum(axis=0)

        bary = self._barycentric_projection(coupling, prev_topics, transported_mass_new)
        unmatched_mask = transported_mass_new < self.min_transport_mass

        return AlignmentResult(
            coupling=coupling,
            prev_weights=prev_weights,
            new_weights=new_weights,
            transported_mass_prev=transported_mass_prev,
            transported_mass_new=transported_mass_new,
            barycentric_projection=bary,
            unmatched_mask=unmatched_mask,
        )

    def merge_topics(
        self,
        new_topics: np.ndarray,
        alignment: AlignmentResult,
    ) -> np.ndarray:
        """Blend new topic embeddings with their aligned counterparts."""

        bary = alignment.barycentric_projection
        smoothing = np.clip(self.smoothing, 0.0, 1.0)
        merged = (1.0 - smoothing) * new_topics + smoothing * bary
        merged[alignment.unmatched_mask] = new_topics[alignment.unmatched_mask]
        return merged

    @staticmethod
    def _normalise_weights(weights: Optional[np.ndarray], size: int) -> np.ndarray:
        if weights is None:
            weights = np.ones(size, dtype=np.float64) / float(size)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim != 1 or weights.shape[0] != size:
                raise ValueError("Weights must be a 1-D array matching the number of topics")
            total = weights.sum()
            if total <= 0:
                raise ValueError("Weights must sum to a positive value")
            weights = weights / total
        return weights

    @staticmethod
    def _barycentric_projection(
        coupling: np.ndarray,
        prev_topics: np.ndarray,
        transported_mass_new: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        bary = np.zeros_like(prev_topics[: coupling.shape[1]])
        for j in range(coupling.shape[1]):
            col = coupling[:, j]
            mass = transported_mass_new[j]
            if mass <= eps:
                bary[j] = prev_topics.mean(axis=0)
            else:
                bary[j] = (col[:, None] * prev_topics).sum(axis=0) / mass
        return bary


@dataclass
class IncrementalState:
    """Book-keeping for incremental training history."""

    topic_embeddings: np.ndarray
    word_embeddings: np.ndarray
    beta: np.ndarray
    vocab: List[str]


class IncrementalTrainer:
    """Manage continual learning for FASTopic using unbalanced OT alignment.

    Parameters
    ----------
    model : FASTopic
        A fitted FASTopic instance that will be updated incrementally.
    aligner : UnbalancedOTAligner, optional
        Custom aligner; defaults to :class:`UnbalancedOTAligner`.
    device : str, optional
        Override device when pushing numpy arrays back into the PyTorch model.
    """

    def __init__(
        self,
        model: FASTopic,
        *,
        aligner: Optional[UnbalancedOTAligner] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.aligner = aligner or UnbalancedOTAligner()
        self.device = device or model.device
        self.state = IncrementalState(
            topic_embeddings=model.topic_embeddings.copy(),
            word_embeddings=model.word_embeddings.copy(),
            beta=model.get_beta().copy(),
            vocab=list(model.vocab) if getattr(model, "vocab", None) is not None else [],
        )

    def update(
        self,
        *,
        cell_embeddings: np.ndarray,
        gene_names: list[str],
        expression_bow: Optional[np.ndarray] = None,
        epochs: int = 100,
        learning_rate: float = 2e-3,
        prev_topic_weights: Optional[np.ndarray] = None,
        new_topic_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Train on a new batch with previous gene embeddings and align topics."""

        prev_topics = self.state.topic_embeddings.copy()

        init_word_embeddings: Optional[Dict[str, np.ndarray]] = None
        if self.state.vocab:
            init_word_embeddings = {
                "embeddings": self.state.word_embeddings.copy(),
                "vocab": self.state.vocab.copy(),
            }

        self.beta = None

        self.model.fit_transform_sc(
            cell_embeddings=cell_embeddings,
            gene_names=gene_names,
            expression_bow=expression_bow,
            epochs=epochs,
            learning_rate=learning_rate,
            init_word_embeddings=init_word_embeddings,
        )

        new_topics = self.model.topic_embeddings.copy()

        alignment = self.aligner.align(
            prev_topics,
            new_topics,
            prev_weights=prev_topic_weights,
            new_weights=new_topic_weights,
        )

        merged_topics = self.aligner.merge_topics(new_topics, alignment)

        with torch.no_grad():
            topic_tensor = torch.as_tensor(merged_topics, device=self.device, dtype=self.model.model.topic_embeddings.dtype)
            self.model.model.topic_embeddings.copy_(topic_tensor)

        self.state = IncrementalState(
            topic_embeddings=self.model.topic_embeddings.copy(),
            word_embeddings=self.model.word_embeddings.copy(),
            beta=self.model.get_beta().copy(),
            vocab=list(self.model.vocab),
        )

        return {
            "transport_plan": alignment.coupling,
            "transported_mass_prev": alignment.transported_mass_prev,
            "transported_mass_new": alignment.transported_mass_new,
            "merged_topics": merged_topics,
            "unmatched_mask": alignment.unmatched_mask,
        }

    def reset_to_state(self, state: Optional[IncrementalState] = None) -> None:
        """Restore the model's topic-related tensors from a stored state."""

        state = state or self.state
        with torch.no_grad():
            topic_tensor = torch.as_tensor(state.topic_embeddings, device=self.device, dtype=self.model.model.topic_embeddings.dtype)
            word_tensor = torch.as_tensor(state.word_embeddings, device=self.device, dtype=self.model.model.word_embeddings.dtype)
            self.model.model.topic_embeddings.copy_(topic_tensor)
            self.model.model.word_embeddings.copy_(word_tensor)
        self.model.vocab = state.vocab.copy()

    def get_state(self) -> IncrementalState:
        """Return a copy of the latest stored incremental state."""

        return dataclasses.replace(
            self.state,
            topic_embeddings=self.state.topic_embeddings.copy(),
            word_embeddings=self.state.word_embeddings.copy(),
            beta=self.state.beta.copy(),
            vocab=self.state.vocab.copy(),
        )
