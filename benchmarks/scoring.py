"""Pure benchmark metric calculations."""

from __future__ import annotations

from typing import Iterable


def score_routes(records: Iterable[tuple[set[str], set[str]]]) -> dict[str, float]:
    """Score ``(required, predicted)`` toolset pairs."""
    pairs = list(records)
    required_total = sum(len(required) for required, _ in pairs)
    predicted_total = sum(len(predicted) for _, predicted in pairs)
    true_positive = sum(len(required & predicted) for required, predicted in pairs)
    exact = sum(required == predicted for required, predicted in pairs)
    return {
        "required_recall": true_positive / required_total if required_total else 1.0,
        "toolset_precision": true_positive / predicted_total if predicted_total else 1.0,
        "exact_set_accuracy": exact / len(pairs) if pairs else 1.0,
    }
