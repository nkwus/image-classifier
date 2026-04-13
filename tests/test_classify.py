from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

import image_classifier.classify as classify_mod
from image_classifier.classify import FOOD_LABELS, _LABEL_BATCH_SIZE, classify_food


class TestClassifyFood:
    """Tests for the zero-shot food classification."""

    @staticmethod
    def _make_frame(width: int = 64, height: int = 48) -> NDArray[np.uint8]:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def setup_method(self) -> None:
        self._orig_classifier = classify_mod._classifier

    def teardown_method(self) -> None:
        classify_mod._classifier = self._orig_classifier

    def test_returns_label_and_confidence(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "pizza", "score": 0.95}]
        classify_mod._classifier = mock_classifier

        label, score = classify_food(self._make_frame())

        assert label == "Pizza"
        assert score == pytest.approx(0.95)

    def test_title_cases_label(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "french fries", "score": 0.87}]
        classify_mod._classifier = mock_classifier

        label, _ = classify_food(self._make_frame())

        assert label == "French Fries"

    def test_passes_candidate_labels(self) -> None:
        from PIL import Image

        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "sushi", "score": 0.91}]
        classify_mod._classifier = mock_classifier

        with patch("image_classifier.usda.load_cached_labels", return_value=None):
            classify_food(self._make_frame())

        # With batching, all FOOD_LABELS are covered across calls
        all_labels: list[str] = []
        for call in mock_classifier.call_args_list:
            assert isinstance(call[0][0], Image.Image)
            all_labels.extend(call[1]["candidate_labels"])
        assert sorted(all_labels) == sorted(FOOD_LABELS)

    def test_uses_cached_usda_labels_when_available(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "broccoli, raw", "score": 0.88}]
        classify_mod._classifier = mock_classifier
        usda_labels = ["broccoli, raw", "strawberries, raw"]

        with patch("image_classifier.usda.load_cached_labels", return_value=usda_labels):
            label, score = classify_food(self._make_frame())

        _, kwargs = mock_classifier.call_args
        assert kwargs["candidate_labels"] == usda_labels
        assert label == "Broccoli, Raw"
        assert score == pytest.approx(0.88)

    def test_batches_large_label_sets(self) -> None:
        """When candidate labels exceed _LABEL_BATCH_SIZE, classify in batches."""
        # Batch 1 returns "apple" at 0.6, batch 2 returns "pizza" at 0.9
        mock_classifier = MagicMock()
        mock_classifier.side_effect = [
            [{"label": "apple", "score": 0.6}],
            [{"label": "pizza", "score": 0.9}],
        ]
        classify_mod._classifier = mock_classifier

        labels = [f"food_{i}" for i in range(_LABEL_BATCH_SIZE + 10)]
        label, score = classify_food(self._make_frame(), candidate_labels=labels)

        assert mock_classifier.call_count == 2
        assert label == "Pizza"
        assert score == pytest.approx(0.9)
        # First batch has _LABEL_BATCH_SIZE items, second has the remainder
        first_batch = mock_classifier.call_args_list[0][1]["candidate_labels"]
        second_batch = mock_classifier.call_args_list[1][1]["candidate_labels"]
        assert len(first_batch) == _LABEL_BATCH_SIZE
        assert len(second_batch) == 10

    def test_custom_candidate_labels(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "banana", "score": 0.80}]
        classify_mod._classifier = mock_classifier

        custom = ["banana", "apple", "orange"]
        label, score = classify_food(self._make_frame(), candidate_labels=custom)

        _, kwargs = mock_classifier.call_args
        assert kwargs["candidate_labels"] == custom
        assert label == "Banana"
        assert score == pytest.approx(0.80)

    def test_lazy_loads_classifier_on_first_call(self) -> None:
        classify_mod._classifier = None
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "ice cream", "score": 0.80}]

        with patch(
            "image_classifier.classify._create_pipeline", return_value=mock_pipeline
        ) as mock_create:
            label, score = classify_food(self._make_frame())

        mock_create.assert_called_once()
        assert label == "Ice Cream"
        assert score == pytest.approx(0.80)

    def test_reuses_cached_classifier(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "steak", "score": 0.70}]
        classify_mod._classifier = mock_classifier

        small_labels = ["steak", "pizza"]
        classify_food(self._make_frame(), candidate_labels=small_labels)
        classify_food(self._make_frame(), candidate_labels=small_labels)

        # One batch call per classify_food invocation (2 labels < batch size)
        assert mock_classifier.call_count == 2


class TestFoodLabels:
    """Tests for the FOOD_LABELS constant."""

    def test_contains_common_foods(self) -> None:
        for food in ("pizza", "sushi", "hamburger", "tacos", "ice cream"):
            assert food in FOOD_LABELS

    def test_all_lowercase(self) -> None:
        for label in FOOD_LABELS:
            assert label == label.lower()


class TestCreatePipeline:
    """Tests for the _create_pipeline helper."""

    def test_calls_transformers_pipeline_with_model(self) -> None:
        mock_pipeline_fn = MagicMock()

        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipeline_fn)}):
            from image_classifier.classify import _create_pipeline

            _create_pipeline()

        mock_pipeline_fn.assert_called_once_with(
            "zero-shot-image-classification",
            model=classify_mod.MODEL_NAME,
        )
