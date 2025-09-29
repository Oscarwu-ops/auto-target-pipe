"""Feature construction utilities."""

from __future__ import annotations

import math
from typing import List, Mapping, Optional, Sequence

from .hypergraph import HyperGraphEncoder
from .network import GeneNetwork


class FeatureExtractor:
    """Combine raw expression, gene network and hypergraph summaries."""

    def __init__(
        self,
        genes: Sequence[str],
        network: Optional[GeneNetwork] = None,
        encoder: Optional[HyperGraphEncoder] = None,
    ):
        if not genes:
            raise ValueError("At least one gene must be supplied")
        self.genes = list(genes)
        self.network = network
        self.encoder = encoder

    @staticmethod
    def _summary(values: Sequence[float]) -> List[float]:
        if not values:
            return [0.0, 0.0, 0.0, 0.0]
        minimum = min(values)
        maximum = max(values)
        mean = sum(values) / len(values)
        variance = 0.0
        for value in values:
            variance += (value - mean) ** 2
        variance /= len(values)
        std_dev = math.sqrt(variance)
        return [mean, std_dev, minimum, maximum]

    def transform_sample(self, expression: Mapping[str, float]) -> List[float]:
        base_values = [float(expression.get(gene, 0.0)) for gene in self.genes]
        features = self._summary(base_values)

        if self.network is not None:
            aggregated = self.network.aggregate_expression(expression)
            aggregated_values = [aggregated.get(gene, 0.0) for gene in self.genes]
            features.extend(self._summary(aggregated_values))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        if self.encoder is not None:
            features.extend(self.encoder.encode(expression))
        return features

    def transform(self, dataset) -> List[List[float]]:
        return [self.transform_sample(sample.expression) for sample in dataset.samples]
