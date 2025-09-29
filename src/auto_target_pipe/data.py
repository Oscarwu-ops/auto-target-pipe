"""Data container utilities for the auto target prediction pipeline.

The goal of this module is to keep the representation of experimental samples
explicit and easy to manipulate.  The implementations rely solely on the
Python standard library so that the project remains lightweight and portable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass(frozen=True)
class SampleRecord:
    """Representation of a single experiment/sample.

    Attributes
    ----------
    name:
        Identifier of the sample.  This is typically a combination of a drug
        and a condition (e.g. cell type) but the pipeline leaves it generic so
        it can be adapted to different experimental setups.
    label:
        Binary label describing whether the sample is associated with a
        desirable target response.  The logistic regression model in the
        pipeline expects ``0`` or ``1``.
    expression:
        Mapping from gene symbol to expression value.
    """

    name: str
    label: int
    expression: Dict[str, float]

    def vector(self, genes: Sequence[str]) -> List[float]:
        """Return the expression values for ``genes`` in the specified order."""

        return [float(self.expression.get(gene, 0.0)) for gene in genes]


class ExpressionDataset:
    """Collection of :class:`SampleRecord` objects with a shared gene space."""

    def __init__(self, genes: Sequence[str], samples: Sequence[SampleRecord]):
        if not genes:
            raise ValueError("At least one gene must be provided")
        self._genes: List[str] = list(genes)
        self._samples: List[SampleRecord] = list(samples)

    @property
    def genes(self) -> List[str]:
        return list(self._genes)

    @property
    def samples(self) -> List[SampleRecord]:
        return list(self._samples)

    @property
    def labels(self) -> List[int]:
        return [sample.label for sample in self._samples]

    def matrix(self) -> List[List[float]]:
        """Return a dense matrix of expression values.

        Each row corresponds to a sample and each column corresponds to a gene
        in the order defined by :pyattr:`genes`.
        """

        return [sample.vector(self._genes) for sample in self._samples]

    @classmethod
    def from_records(cls, records: Iterable[SampleRecord]) -> "ExpressionDataset":
        records = list(records)
        if not records:
            raise ValueError("At least one sample record must be provided")
        genes: Dict[str, None] = {}
        for record in records:
            genes.update({gene: None for gene in record.expression})
        ordered_genes = sorted(genes)
        return cls(ordered_genes, records)

    @classmethod
    def from_csv(cls, path: Path | str) -> "ExpressionDataset":
        """Load an expression dataset from a CSV file.

        The expected format is one sample per row with the following columns:

        ``sample`` (identifier), ``label`` (0 or 1) and one column per gene.
        """

        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty")
            if len(header) < 3:
                raise ValueError("CSV must contain at least sample, label and one gene column")
            genes = header[2:]
            samples: List[SampleRecord] = []
            for row in reader:
                if not row:
                    continue
                sample_name = row[0]
                label = int(row[1])
                expression_values = {
                    gene: float(value) if value else 0.0
                    for gene, value in zip(genes, row[2:])
                }
                samples.append(SampleRecord(sample_name, label, expression_values))
        return cls(genes, samples)

    def iter_vectors(self) -> Iterator[List[float]]:
        """Yield the expression vector for each sample."""

        for sample in self._samples:
            yield sample.vector(self._genes)

    def iter_samples(self) -> Iterator[SampleRecord]:
        """Iterate over the stored samples."""

        return iter(self._samples)
