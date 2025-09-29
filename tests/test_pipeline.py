import unittest

from auto_target_pipe import AutoTargetPipeline, ExpressionNormalizer, GeneInteractionModel, TargetPrioritizer


class ExpressionNormalizerTests(unittest.TestCase):
    def test_normalizes_per_gene(self) -> None:
        dataset = [
            {"G1": 10.0, "G2": 1.0},
            {"G1": 12.0, "G2": 3.0},
            {"G1": 14.0, "G2": 5.0},
        ]
        normalizer = ExpressionNormalizer()
        normalised = normalizer.normalize(dataset)

        self.assertEqual(len(normalised), 3)
        for sample in normalised:
            self.assertSetEqual(set(sample.keys()), {"G1", "G2"})

        g1_values = [sample["G1"] for sample in normalised]
        g2_values = [sample["G2"] for sample in normalised]

        self.assertAlmostEqual(sum(g1_values), 0.0, places=6)
        self.assertAlmostEqual(sum(g2_values), 0.0, places=6)

        # z-scores for linear sequences are predictable when using sample variance
        self.assertAlmostEqual(g1_values[0], -1.0)
        self.assertAlmostEqual(g1_values[1], 0.0)
        self.assertAlmostEqual(g1_values[2], 1.0)

    def test_uses_sample_standard_deviation(self) -> None:
        dataset = [
            {"G1": 2.0, "G2": 4.0},
            {"G1": 4.0, "G2": 8.0},
        ]
        normaliser = ExpressionNormalizer()
        normalised = normaliser.normalize(dataset)

        self.assertAlmostEqual(normalised[0]["G1"], -0.70710678118)
        self.assertAlmostEqual(normalised[1]["G1"], 0.70710678118)
        self.assertAlmostEqual(normalised[0]["G2"], -0.70710678118)
        self.assertAlmostEqual(normalised[1]["G2"], 0.70710678118)

    def test_single_sample_returns_zero_vector(self) -> None:
        dataset = [{"G1": 5.0, "G2": 3.0}]
        normaliser = ExpressionNormalizer()
        normalised = normaliser.normalize(dataset)

        self.assertEqual(normalised, [{"G1": 0.0, "G2": 0.0}])

    def test_rejects_non_numeric_values(self) -> None:
        dataset = [
            {"G1": 1.0, "G2": "high"},
            {"G1": 2.0, "G2": 3.0},
        ]
        normaliser = ExpressionNormalizer()

        with self.assertRaises(TypeError):
            normaliser.normalize(dataset)

    def test_rejects_non_finite_values(self) -> None:
        dataset = [
            {"G1": 1.0, "G2": float("nan")},
            {"G1": 2.0, "G2": 3.0},
        ]

        normaliser = ExpressionNormalizer()

        with self.assertRaises(ValueError):
            normaliser.normalize(dataset)


class GeneInteractionModelTests(unittest.TestCase):
    def test_interaction_strength_uses_correlation(self) -> None:
        dataset = [
            {"G1": 1.0, "G2": 2.0, "G3": 5.0},
            {"G1": 2.0, "G2": 4.0, "G3": 4.5},
            {"G1": 3.0, "G2": 6.0, "G3": 6.0},
            {"G1": 4.0, "G2": 8.0, "G3": 4.2},
        ]
        interaction_model = GeneInteractionModel()
        strengths = interaction_model.interaction_strength(dataset)

        self.assertAlmostEqual(strengths["G1"], strengths["G2"], places=6)
        self.assertGreater(strengths["G1"], strengths["G3"])
        self.assertLess(strengths["G3"], 0.3)

    def test_accepts_pre_normalised_dataset(self) -> None:
        dataset = [
            {"G1": 1.0, "G2": 2.0},
            {"G1": 1.5, "G2": 1.8},
            {"G1": 0.8, "G2": 1.2},
        ]
        normaliser = ExpressionNormalizer()
        normalised = normaliser.normalize(dataset)
        interaction_model = GeneInteractionModel(normalizer=normaliser)

        strengths = interaction_model.interaction_strength(
            dataset,
            normalised_dataset=normalised,
        )

        strengths_default = interaction_model.interaction_strength(dataset)

        self.assertEqual(strengths, strengths_default)


class TargetPrioritizerTests(unittest.TestCase):
    def test_combines_components_and_metadata(self) -> None:
        normalised = [
            {"G1": 1.0, "G2": 0.0},
            {"G1": 1.0, "G2": 0.0},
        ]
        interactions = {"G1": 0.8, "G2": 0.1}
        metadata = {"G1": {"prior": 0.2}, "G2": {"prior": 0.05}}

        prioritizer = TargetPrioritizer(expression_weight=0.7, interaction_weight=0.3)
        predictions = prioritizer.score(normalised, interactions, metadata)

        self.assertEqual(predictions[0].gene, "G1")
        self.assertGreater(predictions[0].score, predictions[1].score)
        self.assertAlmostEqual(predictions[0].metadata["prior"], 0.2)

    def test_allows_empty_metadata_without_error(self) -> None:
        normalised = [
            {"G1": 1.0, "G2": -1.0},
        ]
        interactions = {"G1": 0.5, "G2": 0.5}
        metadata = {"G1": {}, "G2": {}}

        prioritizer = TargetPrioritizer()

        predictions = prioritizer.score(normalised, interactions, metadata)

        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(pred.metadata == metadata[pred.gene] for pred in predictions))

    def test_rejects_non_numeric_metadata(self) -> None:
        normalised = [
            {"G1": 1.0},
            {"G1": -1.0},
        ]

        prioritizer = TargetPrioritizer()

        with self.assertRaises(TypeError):
            prioritizer.score(normalised, {}, {"G1": {"prior": "high"}})

    def test_rejects_non_finite_metadata(self) -> None:
        normalised = [
            {"G1": 1.0},
            {"G1": -1.0},
        ]

        prioritizer = TargetPrioritizer()

        with self.assertRaises(ValueError):
            prioritizer.score(normalised, {}, {"G1": {"prior": float("inf")}})


class AutoTargetPipelineTests(unittest.TestCase):
    def test_pipeline_end_to_end(self) -> None:
        dataset = [
            {"G1": 5.0, "G2": 3.0, "G3": 1.0},
            {"G1": 4.0, "G2": 2.5, "G3": 1.5},
            {"G1": 6.0, "G2": 2.0, "G3": 2.0},
        ]
        metadata = {"G3": {"clinical": 0.1}}

        pipeline = AutoTargetPipeline()
        predictions = pipeline.run(dataset, metadata, top_k=2)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0].gene, "G3")
        self.assertTrue(all(pred.score >= predictions[-1].score for pred in predictions))

    def test_rejects_mismatched_gene_sets(self) -> None:
        dataset = [
            {"G1": 1.0, "G2": 2.0},
            {"G1": 1.5, "G3": 2.5},
        ]
        pipeline = AutoTargetPipeline()
        with self.assertRaises(ValueError):
            pipeline.run(dataset)

    def test_reuses_normalised_dataset_for_interactions(self) -> None:
        dataset = [
            {"G1": 2.0, "G2": 4.0},
            {"G1": 3.0, "G2": 5.0},
            {"G1": 4.0, "G2": 6.0},
        ]

        class RecordingNormalizer(ExpressionNormalizer):
            def __init__(self) -> None:
                super().__init__()
                self.last_output = None

            def normalize(self, dataset):  # type: ignore[override]
                result = super().normalize(dataset)
                self.last_output = result
                return result

        class RecordingInteractionModel(GeneInteractionModel):
            def __init__(self) -> None:
                super().__init__()
                self.normalizer = None  # allow pipeline to inject shared normalizer
                self.received = None
                self.calls = 0

            def interaction_strength(self, dataset, *, normalised_dataset=None):  # type: ignore[override]
                self.calls += 1
                self.received = normalised_dataset
                return super().interaction_strength(
                    dataset,
                    normalised_dataset=normalised_dataset,
                )

        normalizer = RecordingNormalizer()
        interaction_model = RecordingInteractionModel()
        pipeline = AutoTargetPipeline(
            normalizer=normalizer,
            interaction_model=interaction_model,
        )

        pipeline.run(dataset)

        self.assertIs(interaction_model.received, normalizer.last_output)
        self.assertEqual(interaction_model.calls, 1)


if __name__ == "__main__":
    unittest.main()
