"""Food classification using zero-shot image classification (SigLIP)."""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

MODEL_NAME = "google/siglip-so400m-patch14-384"
_LABEL_BATCH_SIZE = 100  # max labels per forward pass to avoid GPU OOM

# Food-101 categories, human-readable.
FOOD_LABELS: list[str] = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio",
    "beef tartare", "beet salad", "beignets", "bibimbap",
    "bread pudding", "breakfast burrito", "bruschetta", "caesar salad",
    "cannoli", "caprese salad", "carrot cake", "ceviche",
    "cheese plate", "cheesecake", "chicken curry", "chicken quesadilla",
    "chicken wings", "chocolate cake", "chocolate mousse",
    "churros", "clam chowder", "club sandwich", "crab cakes",
    "creme brulee", "croque madame", "cup cakes", "deviled eggs",
    "donuts", "dumplings", "edamame", "eggs benedict",
    "escargots", "falafel", "filet mignon", "fish and chips",
    "foie gras", "french fries", "french onion soup", "french toast",
    "fried calamari", "fried rice", "frozen yogurt", "garlic bread",
    "gnocchi", "greek salad", "grilled cheese sandwich",
    "grilled salmon", "guacamole", "gyoza", "hamburger",
    "hot and sour soup", "hot dog", "huevos rancheros", "hummus",
    "ice cream", "lasagna", "lobster bisque", "lobster roll sandwich",
    "macaroni and cheese", "macarons", "miso soup", "mussels",
    "nachos", "omelette", "onion rings", "oysters", "pad thai",
    "paella", "pancakes", "panna cotta", "peking duck",
    "pho", "pizza", "pork chop", "poutine", "prime rib",
    "pulled pork sandwich", "ramen", "ravioli", "red velvet cake",
    "risotto", "samosa", "sashimi", "scallops",
    "seaweed salad", "shrimp and grits", "spaghetti bolognese",
    "spaghetti carbonara", "spring rolls", "steak",
    "strawberry shortcake", "sushi", "tacos", "takoyaki",
    "tiramisu", "tuna tartare", "waffles", 'soda', 'coffee', 'tea', 'cookie', 'werthers original',
    "mandarin orange", "grapefruit", "lemon", "lime", "orange", "pear", "pineapple", 'sweet potato',
]

_classifier: Any = None


def _create_pipeline() -> Any:
    """Import transformers and build the zero-shot image-classification pipeline."""
    from transformers import pipeline

    return pipeline(
        "zero-shot-image-classification",
        model=MODEL_NAME,
    )


def _get_classifier() -> Any:
    """Return the zero-shot classification pipeline, loading it on first call."""
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        print("Loading food classification model…", file=sys.stderr)
        _classifier = _create_pipeline()
    return _classifier


def classify_food(
    image: NDArray[np.uint8],
    candidate_labels: list[str] | None = None,
) -> tuple[str, float]:
    """Classify a BGR image as a food item using zero-shot classification.

    *candidate_labels* defaults to :data:`FOOD_LABELS` (Food-101 categories).

    Returns ``(label, confidence)`` where *label* is a human-readable food
    name and *confidence* is a float in [0, 1].
    """
    if candidate_labels is None:
        from image_classifier.usda import load_cached_labels

        candidate_labels = load_cached_labels() or FOOD_LABELS
    classifier = _get_classifier()
    rgb: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Process labels in batches to avoid GPU out-of-memory errors when the
    # candidate set is large (e.g. thousands of USDA labels).
    from tqdm import tqdm

    total_batches = math.ceil(len(candidate_labels) / _LABEL_BATCH_SIZE)
    best_label = ""
    best_score = -1.0

    # Suppress the "using the pipelines sequentially on GPU" warning —
    # we are batching label chunks, not processing multiple images.
    # Transformers emits this via logging, not warnings.warn().
    _pipelines_logger = logging.getLogger("transformers.pipelines.base")
    _prev_level = _pipelines_logger.level
    _pipelines_logger.setLevel(logging.ERROR)
    try:
        for i in tqdm(
            range(0, len(candidate_labels), _LABEL_BATCH_SIZE),
            total=total_batches,
            desc="Classifying",
            unit="batch",
            file=sys.stderr,
        ):
            batch = candidate_labels[i : i + _LABEL_BATCH_SIZE]
            results: list[dict[str, Any]] = classifier(
                pil_image, candidate_labels=batch
            )
            top = results[0]
            if float(top["score"]) > best_score:
                best_label = str(top["label"])
                best_score = float(top["score"])
    finally:
        _pipelines_logger.setLevel(_prev_level)

    return best_label.title(), best_score
