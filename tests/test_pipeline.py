from __future__ import annotations

import math
from pathlib import Path

from auto_target_pipe import (
    AutoTargetPipeline,
    ExpressionDataset,
    GeneNetwork,
    HyperGraph,
    HyperGraphEncoder,
    PipelineConfig,
    SampleRecord,
)
from auto_target_pipe.features import FeatureExtractor


def build_dataset() -> ExpressionDataset:
    positives = [
        SampleRecord("p1", 1, {"G1": 3.1, "G2": 2.8, "G3": 3.0, "G4": 2.6}),
        SampleRecord("p2", 1, {"G1": 3.3, "G2": 2.9, "G3": 3.2, "G4": 2.4}),
        SampleRecord("p3", 1, {"G1": 3.2, "G2": 2.7, "G3": 3.1, "G4": 2.5}),
    ]
    negatives = [
        SampleRecord("n1", 0, {"G1": 0.3, "G2": 0.7, "G3": 0.4, "G4": 0.6}),
        SampleRecord("n2", 0, {"G1": 0.2, "G2": 0.6, "G3": 0.3, "G4": 0.5}),
        SampleRecord("n3", 0, {"G1": 0.1, "G2": 0.8, "G3": 0.2, "G4": 0.4}),
    ]
    return ExpressionDataset.from_records(positives + negatives)


def test_dataset_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "expression.csv"
    csv_path.write_text(
        "sample,label,G1,G2,G3,G4\n"
        "s1,1,1.0,2.0,3.0,4.0\n"
        "s2,0,0.1,0.2,0.3,0.4\n",
        encoding="utf8",
    )
    dataset = ExpressionDataset.from_csv(csv_path)
    assert dataset.genes == ["G1", "G2", "G3", "G4"]
    assert dataset.labels == [1, 0]
    assert dataset.matrix() == [[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]]


def test_feature_extraction_shapes() -> None:
    dataset = build_dataset()
    network = GeneNetwork.from_edges([("G1", "G2"), ("G3", "G4")])
    hypergraph = HyperGraph.from_iterable([["G1", "G3"], ["G2", "G4"]])
    encoder = HyperGraphEncoder(hypergraph, latent_dim=3, seed=7)
    extractor = FeatureExtractor(dataset.genes, network=network, encoder=encoder)
    sample_features = extractor.transform_sample(dataset.samples[0].expression)
    assert len(sample_features) == 4 + 4 + 3
    assert not math.isnan(sample_features[0])


def test_pipeline_training_and_prediction() -> None:
    dataset = build_dataset()
    network = GeneNetwork.from_edges([("G1", "G2"), ("G2", "G3"), ("G3", "G4")])
    hypergraph = HyperGraph.from_iterable([["G1", "G3", "G4"], ["G2", "G3"]])
    encoder = HyperGraphEncoder(hypergraph, latent_dim=2, seed=11)
    pipeline = AutoTargetPipeline(
        dataset.genes,
        network=network,
        encoder=encoder,
        config=PipelineConfig(learning_rate=0.5, epochs=500, l2_penalty=0.0),
    )
    pipeline.fit(dataset)
    scores = pipeline.predict_scores(dataset)
    predictions = pipeline.predict_labels(dataset, threshold=0.5)
    assert len(scores) == len(dataset.samples)
    assert all(0.0 <= score <= 1.0 for score in scores)
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, dataset.labels)) / len(predictions)
    assert accuracy >= 0.83
