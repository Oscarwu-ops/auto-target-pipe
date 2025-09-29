"""Simple, fully self contained auto target prioritisation pipeline.

This module does not re-implement the original scDrugPrio, Genes2genes or
hyperG-VAE projects.  Instead it provides a light-weight, fully documented
approximation that captures the data flow between their conceptual stages so the
project can be prototyped and tested locally.

The pipeline operates in three phases:

``ExpressionNormalizer``
    Performs per gene z-score normalisation across the supplied samples.  This
    acts as a cheap stand in for the feature scaling done in the scDrugPrio
    preprocessing frame.

``GeneInteractionModel``
    Builds a simple symmetric interaction matrix using Pearson correlation
    between genes across the input samples.  The matrix is then collapsed into a
    per gene interaction strength score by computing the mean absolute
    correlation per gene.  This approximates the role of the Genes2genes core
    which learns relationships between genes.

``TargetPrioritizer``
    Combines the normalised expression values and the interaction scores into a
    final prioritisation score.  The scoring is intentionally transparent:

    .. math::

        score_g = w_1 * mean(|z_g|) + w_2 * interaction_g + metadata_g

    where ``metadata_g`` is an optional prior coming from domain knowledge.  In
    the real project this stage would correspond to the hyperG-VAE component
    that refines gene targets using generative priors.

The :class:`AutoTargetPipeline` orchestrates these building blocks and returns a
list of :class:`TargetPrediction` objects sorted by their score.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


Number = float
ExpressionVector = Mapping[str, Number]
ExpressionDataset = Sequence[ExpressionVector]
GeneMetadata = Mapping[str, Mapping[str, Number]]


def _ensure_expression_keys_consistency(dataset: ExpressionDataset) -> List[str]:
    """Validate that every sample exposes the same set of gene identifiers.

    Parameters
    ----------
    dataset:
        Iterable collection of sample expression vectors.

    Returns
    -------
    list of str
        Ordered list of gene identifiers that appear in the dataset.

    Raises
    ------
    ValueError
        If samples disagree on the set of genes they contain.
    """

    iterator = iter(dataset)
    try:
        first_sample = next(iterator)
    except StopIteration as exc:  # pragma: no cover - protected by explicit check
        raise ValueError("expression dataset is empty") from exc

    gene_ids = sorted(first_sample.keys())
    gene_set = set(gene_ids)

    _validate_expression_values(first_sample, 0)

    for sample_idx, sample in enumerate(iterator, start=1):
        if set(sample.keys()) != gene_set:
            raise ValueError(
                "all samples must expose the same gene identifiers"
            )
        _validate_expression_values(sample, sample_idx)

    return gene_ids


def _validate_expression_values(sample: Mapping[str, Number], sample_idx: int) -> None:
    for gene, value in sample.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"expression value for gene '{gene}' in sample {sample_idx} must be a number"
            )
        if not isfinite(value):
            raise ValueError(
                f"expression value for gene '{gene}' in sample {sample_idx} must be finite"
            )


def _validate_metadata_values(metadata: Mapping[str, Number], gene: str) -> Iterable[Number]:
    """Yield metadata values after validating they are numeric and finite."""

    for key, value in metadata.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"metadata value '{key}' for gene '{gene}' must be a number"
            )
        if not isfinite(value):
            raise ValueError(
                f"metadata value '{key}' for gene '{gene}' must be finite"
            )
        yield value


class ExpressionNormalizer:
    """Normalise gene expression across samples using a z-score transformation."""

    def __init__(self, ddof: int = 1) -> None:
        """Create a normaliser.

        Parameters
        ----------
        ddof:
            Delta degrees of freedom used when computing the sample standard
            deviation.  ``ddof=1`` (the default) yields the unbiased estimator
            for finite samples, while ``ddof=0`` switches to population
            statistics.
        """
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        self.ddof = ddof

    def normalize(self, dataset: ExpressionDataset) -> List[Dict[str, Number]]:
        genes = _ensure_expression_keys_consistency(dataset)
        normalised: List[Dict[str, Number]] = []

        # compute per gene statistics
        per_gene_values: Dict[str, List[Number]] = {gene: [] for gene in genes}
        for sample in dataset:
            for gene, value in sample.items():
                per_gene_values[gene].append(value)

        per_gene_means: Dict[str, Number] = {
            gene: mean(values) for gene, values in per_gene_values.items()
        }
        per_gene_std: Dict[str, Number] = {}
        for gene, values in per_gene_values.items():
            mu = per_gene_means[gene]
            squared_diffs = [(value - mu) ** 2 for value in values]
            if len(values) - self.ddof <= 0:
                per_gene_std[gene] = 0.0
            else:
                per_gene_std[gene] = sqrt(
                    sum(squared_diffs) / (len(values) - self.ddof)
                )

        for sample in dataset:
            z_sample: Dict[str, Number] = {}
            for gene in genes:
                sigma = per_gene_std[gene]
                if sigma == 0:
                    z_value = 0.0
                else:
                    z_value = (sample[gene] - per_gene_means[gene]) / sigma
                z_sample[gene] = z_value
            normalised.append(z_sample)

        return normalised


class GeneInteractionModel:
    """Compute a simple correlation based interaction score for each gene."""

    def __init__(self, normalizer: Optional[ExpressionNormalizer] = None) -> None:
        self.normalizer = normalizer or ExpressionNormalizer()

    def interaction_strength(
        self,
        dataset: ExpressionDataset,
        *,
        normalised_dataset: Optional[ExpressionDataset] = None,
    ) -> Dict[str, Number]:
        """Return per gene interaction strengths.

        Parameters
        ----------
        dataset:
            Raw expression dataset.  Used for validation when ``normalised_dataset``
            is provided, and for on-demand normalisation otherwise.
        normalised_dataset:
            Optional pre-normalised dataset.  Passing this allows callers to
            reuse existing computations instead of repeating them here.
        """
        if normalised_dataset is None:
            normalised_dataset = self.normalizer.normalize(dataset)
        genes = _ensure_expression_keys_consistency(normalised_dataset)

        # Build correlation matrix; we only need mean absolute correlation per gene
        per_gene_vectors: Dict[str, List[Number]] = {gene: [] for gene in genes}
        for sample in normalised_dataset:
            for gene, value in sample.items():
                per_gene_vectors[gene].append(value)

        strengths: Dict[str, Number] = {}
        for idx, gene in enumerate(genes):
            correlations: List[Number] = []
            vector_a = per_gene_vectors[gene]
            for jdx, other_gene in enumerate(genes):
                if idx == jdx:
                    continue
                vector_b = per_gene_vectors[other_gene]
                correlations.append(_pearson_correlation(vector_a, vector_b))
            if correlations:
                strengths[gene] = mean(abs(corr) for corr in correlations)
            else:
                strengths[gene] = 0.0
        return strengths


def _pearson_correlation(vector_a: Sequence[Number], vector_b: Sequence[Number]) -> Number:
    if len(vector_a) != len(vector_b):  # pragma: no cover - programming error
        raise ValueError("vectors must have identical length")
    n = len(vector_a)
    if n == 0:  # pragma: no cover - unreachable due to validation
        raise ValueError("vectors must be non-empty")
    mean_a = mean(vector_a)
    mean_b = mean(vector_b)
    numerator = 0.0
    denom_a = 0.0
    denom_b = 0.0
    for a, b in zip(vector_a, vector_b):
        da = a - mean_a
        db = b - mean_b
        numerator += da * db
        denom_a += da * da
        denom_b += db * db
    denominator = (denom_a * denom_b) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class TargetPrediction:
    gene: str
    score: Number
    metadata: Mapping[str, Number]


class TargetPrioritizer:
    """Combine scores from each stage into an interpretable target ranking."""

    def __init__(self, expression_weight: Number = 0.6, interaction_weight: Number = 0.4):
        if expression_weight < 0 or interaction_weight < 0:
            raise ValueError("weights must be non-negative")
        total = expression_weight + interaction_weight
        if total == 0:
            raise ValueError("at least one weight must be non-zero")
        self.expression_weight = expression_weight / total
        self.interaction_weight = interaction_weight / total

    def score(
        self,
        normalised_dataset: ExpressionDataset,
        interaction_strengths: Mapping[str, Number],
        gene_metadata: Optional[GeneMetadata] = None,
    ) -> List[TargetPrediction]:
        genes = _ensure_expression_keys_consistency(normalised_dataset)
        gene_metadata = gene_metadata or {}

        per_gene_values: Dict[str, List[Number]] = {gene: [] for gene in genes}
        for sample in normalised_dataset:
            for gene, value in sample.items():
                per_gene_values[gene].append(abs(value))

        predictions: List[TargetPrediction] = []
        for gene in genes:
            expression_component = mean(per_gene_values[gene])
            interaction_component = interaction_strengths.get(gene, 0.0)
            metadata_mapping = gene_metadata.get(gene)
            metadata_component = 0.0
            if metadata_mapping:
                validated_values = list(_validate_metadata_values(metadata_mapping, gene))
                if validated_values:
                    metadata_component = mean(validated_values)
            score = (
                self.expression_weight * expression_component
                + self.interaction_weight * interaction_component
                + metadata_component
            )
            predictions.append(
                TargetPrediction(
                    gene=gene,
                    score=score,
                    metadata=gene_metadata.get(gene, {}),
                )
            )

        predictions.sort(key=lambda pred: pred.score, reverse=True)
        return predictions


class AutoTargetPipeline:
    """Orchestrate the full target prioritisation workflow."""

    def __init__(
        self,
        normalizer: Optional[ExpressionNormalizer] = None,
        interaction_model: Optional[GeneInteractionModel] = None,
        prioritizer: Optional[TargetPrioritizer] = None,
    ) -> None:
        self.normalizer = normalizer or ExpressionNormalizer()
        if interaction_model is None:
            self.interaction_model = GeneInteractionModel(normalizer=self.normalizer)
        else:
            self.interaction_model = interaction_model
            if getattr(self.interaction_model, "normalizer", None) is None:
                self.interaction_model.normalizer = self.normalizer
        self.prioritizer = prioritizer or TargetPrioritizer()

    def run(
        self,
        dataset: ExpressionDataset,
        gene_metadata: Optional[GeneMetadata] = None,
        top_k: Optional[int] = None,
    ) -> List[TargetPrediction]:
        if not dataset:
            raise ValueError("dataset must not be empty")

        normalised = self.normalizer.normalize(dataset)
        interactions = self.interaction_model.interaction_strength(
            dataset,
            normalised_dataset=normalised,
        )
        predictions = self.prioritizer.score(normalised, interactions, gene_metadata)

        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be a positive integer")
            return predictions[:top_k]
        return predictions
