# auto-target-pipe

A compact, fully self-contained prototype of an automated drug target
prioritisation pipeline inspired by scDrugPrio, Genes2genes, and hyperG-VAE.  The
repository focuses on approachability: everything is implemented in pure Python
with no external dependencies so experiments can run in constrained environments.

## Features

- **Expression normalisation** – z-score scaling per gene across samples using
  sample standard deviation by default.
- **Gene interaction scoring** – correlation-based interaction strength derived
  from the normalised profiles without recomputing intermediate results.
- **Transparent target ranking** – configurable weights combine expression,
  interaction, and metadata priors into interpretable scores, with validation to
  ensure priors are numeric and finite.
- **Strict input validation** – catches non-numeric or non-finite expression
  values early to prevent subtle downstream issues.
- **Unit tests** – cover all stages of the workflow for easy regression testing.

## Quick start

```python
from auto_target_pipe import AutoTargetPipeline

samples = [
    {"G1": 5.0, "G2": 3.0, "G3": 1.0},
    {"G1": 4.0, "G2": 2.5, "G3": 1.5},
    {"G1": 6.0, "G2": 2.0, "G3": 2.0},
]
metadata = {"G3": {"clinical": 0.4}}

pipeline = AutoTargetPipeline()
for prediction in pipeline.run(samples, metadata, top_k=2):
    print(prediction.gene, prediction.score)
```

## Running the tests

```bash
python -m unittest discover -s tests
```
