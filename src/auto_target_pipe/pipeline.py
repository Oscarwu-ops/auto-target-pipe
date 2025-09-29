"""High-level orchestration of the auto target prediction workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data import ExpressionDataset
from .features import FeatureExtractor
from .hypergraph import HyperGraphEncoder
from .model import LogisticRegression
from .network import GeneNetwork


@dataclass
class PipelineConfig:
    learning_rate: float = 0.1
    epochs: int = 300
    l2_penalty: float = 0.01


class AutoTargetPipeline:
    """Combine feature extraction and model training into a single interface."""

    def __init__(
        self,
        genes,
        network: Optional[GeneNetwork] = None,
        encoder: Optional[HyperGraphEncoder] = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.extractor = FeatureExtractor(genes, network=network, encoder=encoder)
        self.model = LogisticRegression(
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            l2_penalty=self.config.l2_penalty,
        )
        self._fitted = False

    def fit(self, dataset: ExpressionDataset) -> None:
        features = self.extractor.transform(dataset)
        self.model.fit(features, dataset.labels)
        self._fitted = True

    def predict_scores(self, dataset: ExpressionDataset) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        features = self.extractor.transform(dataset)
        return self.model.predict_proba(features)

    def predict_labels(self, dataset: ExpressionDataset, threshold: float = 0.5) -> list[int]:
        scores = self.predict_scores(dataset)
        return [1 if score >= threshold else 0 for score in scores]
