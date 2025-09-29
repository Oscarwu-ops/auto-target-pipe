"""Minimal gene-gene interaction network utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


class GeneNetwork:
    """Representation of an undirected gene interaction graph."""

    def __init__(self, adjacency: Mapping[str, Sequence[Tuple[str, float]]]):
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {
            gene: list(neighbours) for gene, neighbours in adjacency.items()
        }
        for gene, neighbours in list(self._adjacency.items()):
            for neighbour, weight in neighbours:
                if neighbour not in self._adjacency:
                    self._adjacency[neighbour] = []
                if gene not in {g for g, _ in self._adjacency[neighbour]}:
                    self._adjacency[neighbour].append((gene, weight))

    @classmethod
    def from_edges(
        cls, edges: Iterable[Tuple[str, str, float]] | Iterable[Tuple[str, str]]
    ) -> "GeneNetwork":
        """Construct the network from an edge list.

        If weights are omitted they default to ``1.0``.
        """

        adjacency: MutableMapping[str, List[Tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            if len(edge) == 2:
                left, right = edge  # type: ignore[misc]
                weight = 1.0
            else:
                left, right, weight = edge  # type: ignore[misc]
            adjacency[left].append((right, float(weight)))
            adjacency[right].append((left, float(weight)))
        return cls(adjacency)

    def aggregate_expression(self, expression: Mapping[str, float]) -> Dict[str, float]:
        """Aggregate neighbour expression for each gene.

        The aggregation is a weighted average of a gene's neighbours.  Genes
        without neighbours simply reuse their own expression value.
        """

        aggregated: Dict[str, float] = {}
        for gene, neighbours in self._adjacency.items():
            if not neighbours:
                aggregated[gene] = float(expression.get(gene, 0.0))
                continue
            total_weight = sum(weight for _, weight in neighbours)
            if total_weight == 0.0:
                aggregated[gene] = 0.0
                continue
            score = 0.0
            for neighbour, weight in neighbours:
                score += float(expression.get(neighbour, 0.0)) * weight
            aggregated[gene] = score / total_weight
        return aggregated

    def degree(self, gene: str) -> int:
        return len(self._adjacency.get(gene, ()))

    def genes(self) -> List[str]:
        return sorted(self._adjacency)
