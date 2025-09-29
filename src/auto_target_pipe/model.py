"""A lightweight implementation of logistic regression using gradient descent."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


class LogisticRegression:
    """Binary logistic regression implemented with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 200,
        l2_penalty: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if l2_penalty < 0:
            raise ValueError("l2_penalty cannot be negative")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self._weights: List[float] | None = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x < -50:
            return 1e-22
        if x > 50:
            return 1.0 - 1e-22
        return 1.0 / (1.0 + math.exp(-x))

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        if not features:
            raise ValueError("No training data supplied")
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")
        num_features = len(features[0])
        self._weights = [0.0] * (num_features + 1)
        for epoch in range(self.epochs):
            gradients = [0.0] * (num_features + 1)
            for vector, label in zip(features, labels):
                if len(vector) != num_features:
                    raise ValueError("Inconsistent feature vector size")
                activation = self._weights[-1]  # bias
                for weight, value in zip(self._weights[:-1], vector):
                    activation += weight * value
                probability = self._sigmoid(activation)
                error = probability - float(label)
                for index, value in enumerate(vector):
                    gradients[index] += error * value
                gradients[-1] += error
            for index in range(num_features):
                gradients[index] = gradients[index] / len(features) + self.l2_penalty * self._weights[index]
            gradients[-1] = gradients[-1] / len(features) + self.l2_penalty * self._weights[-1]
            for index in range(num_features + 1):
                self._weights[index] -= self.learning_rate * gradients[index]

    def _check_fitted(self) -> List[float]:
        if self._weights is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._weights

    def predict_proba(self, features: Sequence[Sequence[float]]) -> List[float]:
        weights = self._check_fitted()
        probabilities: List[float] = []
        for vector in features:
            activation = weights[-1]
            for weight, value in zip(weights[:-1], vector):
                activation += weight * value
            probabilities.append(self._sigmoid(activation))
        return probabilities

    def predict(self, features: Sequence[Sequence[float]], threshold: float = 0.5) -> List[int]:
        return [1 if probability >= threshold else 0 for probability in self.predict_proba(features)]
