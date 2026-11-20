
"""Unit tests for the SignSight model adapter."""

import unittest
from ui.model_adapter import to_prediction


class ModelAdapterTests(unittest.TestCase):

    def test_correct_prediction(self):
        result = to_prediction(
            [0.92, 0.05, 0.03],
            ["A", "B", "C"]
        )

        self.assertEqual(result["label"], "A")
        self.assertAlmostEqual(result["confidence"], 0.92)
        self.assertTrue(result["hand_visible"])

    def test_top_three_sorted(self):
        result = to_prediction(
            [0.10, 0.70, 0.20],
            ["A", "B", "C"]
        )

        self.assertEqual(
            [item[0] for item in result["top3"]],
            ["B", "C", "A"]
        )

    def test_no_hand_visible(self):
        result = to_prediction(
            [],
            [],
            hand_visible=False
        )

        self.assertFalse(result["hand_visible"])
        self.assertIsNone(result["label"])
        self.assertEqual(result["top3"], [])

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            to_prediction(
                [0.5, 0.5],
                ["A"]
            )

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            to_prediction(
                [-0.2, 1.2],
                ["A", "B"]
            )

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            to_prediction(
                [0.2, 0.3],
                ["A", "B"]
            )

    def test_reject_infinity(self):
        with self.assertRaises(ValueError):
            to_prediction(
                [float("inf"), 0.5],
                ["A", "B"]
            )

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            to_prediction(
                [float("nan"), 0.5],
                ["A", "B"]
            )


if __name__ == "__main__":
    unittest.main()
