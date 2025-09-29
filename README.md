# auto-target-pipe

A lightweight, fully self-contained prototype of an automated therapeutic
target prediction pipeline.  The implementation follows the high-level idea of
combining transcriptomic expression signals with relational knowledge from
Genes2Genes-like interaction graphs and hypergraph-inspired embeddings similar
to hyperG-VAE.

The repository intentionally avoids heavy third-party dependencies so that the
core ideas can run in constrained environments (such as this challenge).  The
pipeline therefore focuses on easy-to-audit Python code backed by unit tests.

## Project layout

```
src/auto_target_pipe/
├── data.py           # Sample/experiment data containers
├── features.py       # Feature extraction orchestration
├── hypergraph.py     # Hypergraph utilities and encoder
├── model.py          # Pure-Python logistic regression
├── network.py        # Gene-gene interaction aggregation
└── pipeline.py       # High-level AutoTargetPipeline facade
```

Unit tests live under `tests/` and validate the main workflow.

## Usage

1. Prepare an expression CSV with columns `sample`, `label` and one column per
gene.
2. Define a gene network edge list and hypergraph specification for the
pathways of interest.
3. Train and evaluate the pipeline as shown below.

```python
from auto_target_pipe import (
    AutoTargetPipeline,
    ExpressionDataset,
    GeneNetwork,
    HyperGraph,
    HyperGraphEncoder,
    PipelineConfig,
)

# Load data
expression = ExpressionDataset.from_csv("expression.csv")
network = GeneNetwork.from_edges([("G1", "G2"), ("G2", "G3")])
hypergraph = HyperGraph.from_iterable([["G1", "G3"], ["G2", "G4"]])
encoder = HyperGraphEncoder(hypergraph, latent_dim=2, seed=42)

# Train the model
pipeline = AutoTargetPipeline(
    expression.genes,
    network=network,
    encoder=encoder,
    config=PipelineConfig(learning_rate=0.3, epochs=400, l2_penalty=0.0),
)
pipeline.fit(expression)

# Score the training cohort (or a held-out dataset)
probabilities = pipeline.predict_scores(expression)
print(probabilities)
```

## Running tests

```bash
python -m pytest
```

The tests exercise CSV ingestion, feature extraction and end-to-end training
with a synthetic dataset that mimics a Genes2Genes + hyperG-VAE inspired setup.
