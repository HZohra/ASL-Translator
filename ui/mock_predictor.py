"""Temporary, deterministic predictions; no neural network is loaded."""

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEQUENCE = "HELLO WORLD"


def mock_prediction(index):
    """Return a demo prediction with top-3 guesses (or no hand for spaces)."""
    sign = SEQUENCE[index % len(SEQUENCE)]
    if sign == " ":
        return {"hand_visible": False, "label": None, "confidence": 0.0, "top3": []}
    others = [x for x in ALPHABET if x != sign]
    return {
        "hand_visible": True,
        "label": sign,
        "confidence": 0.94,
        "top3": [(sign, 0.94), (others[0], 0.04), (others[1], 0.02)],
    }
