
"""Convert future model probabilities into SignSight predictions."""

import math


def to_prediction(probabilities, labels, hand_visible=True):
    """
    Convert future JAX/NumPy probabilities into SignSight format.

    probabilities: probabilities between 0 and 1
    labels: corresponding ASL letters or signs

    Returns the format expected by TranslationController.
    """

    if not hand_visible:
        return {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": []
        }

    if len(probabilities) != len(labels) or not labels:
        raise ValueError(
            "Probabilities and labels must have equal, "
            "nonzero lengths."
        )

    scores = [float(p) for p in probabilities]

    if any(
        not math.isfinite(p) or p < 0 or p > 1
        for p in scores
    ):
        raise ValueError(
            "Probabilities must be finite values from 0 to 1."
        )

    if abs(sum(scores) - 1.0) > 0.01:
        raise ValueError(
            "Probabilities must sum to approximately 1. "
            "Pass softmax probabilities, not logits."
        )

    ranked = sorted(
        zip(labels, scores),
        key=lambda item: item[1],
        reverse=True
    )

    name, confidence = ranked[0]

    return {
        "hand_visible": True,
        "label": str(name),
        "confidence": confidence,
        "top3": [
            [str(label), score]
            for label, score in ranked[:3]
        ]
    }
