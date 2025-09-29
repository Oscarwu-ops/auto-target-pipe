"""Simple hypergraph feature construction inspired by hyperG-VAE."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Set


@dataclass(frozen=True)
class HyperGraph:
    """Representation of a hypergraph using sets of genes as hyperedges."""

    hyperedges: Sequence[Set[str]]

    @classmethod
    def from_iterable(cls, hyperedges: Iterable[Iterable[str]]) -> "HyperGraph":
        edges = [frozenset(edge) for edge in hyperedges if edge]
        if not edges:
            raise ValueError("At least one hyperedge must be provided")
        return cls(edges)

    def aggregate(self, expression: Mapping[str, float]) -> List[float]:
        """Compute the mean expression per hyperedge."""

        features: List[float] = []
        for edge in self.hyperedges:
            if not edge:
                features.append(0.0)
                continue
            total = 0.0
            for gene in edge:
                total += float(expression.get(gene, 0.0))
            features.append(total / len(edge))
        return features


class HyperGraphEncoder:
    """Project hypergraph features to a compact latent representation."""

    def __init__(self, hypergraph: HyperGraph, latent_dim: int = 4, seed: int = 13):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.hypergraph = hypergraph
        self.latent_dim = latent_dim
        self._projection = self._create_projection(len(hypergraph.hyperedges), latent_dim, seed)

    @staticmethod
    def _create_projection(num_features: int, latent_dim: int, seed: int) -> List[List[float]]:
        rng = random.Random(seed)
        projection: List[List[float]] = []
        for _ in range(latent_dim):
            row = []
            for _ in range(num_features):
                row.append(rng.uniform(-1.0, 1.0))
            projection.append(row)
        return projection

    def encode(self, expression: Mapping[str, float]) -> List[float]:
        raw_features = self.hypergraph.aggregate(expression)
        if not raw_features:
            return [0.0] * self.latent_dim
        encoded: List[float] = []
        for row in self._projection:
            value = 0.0
            for weight, feature in zip(row, raw_features):
                value += weight * feature
            encoded.append(value / math.sqrt(len(raw_features)))
        return encoded
