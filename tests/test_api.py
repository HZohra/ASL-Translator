
"""Tests for SignSight's Python API logic.

No webcam, JAX model, or OpenCV required.
"""

import unittest

from backend import server


class SignSightAPITests(unittest.TestCase):

    def setUp(self):
        # Give each test a fresh controller.
        server.controller = server.TranslationController()
        server.demo_index = 0
        server.prediction = {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": [],
        }

    def test_start_translation(self):
        result = server.handle_action({
            "action": "start"
        })
        self.assertTrue(result["running"])

    def test_stop_translation(self):
        server.handle_action({"action": "start"})
        result = server.handle_action({
            "action": "stop"
        })
        self.assertFalse(result["running"])

    def test_confidence_settings(self):
        result = server.handle_action({
            "action": "settings",
            "threshold": 95,
            "holdFrames": 4,
        })
        self.assertEqual(result["threshold"], 95)
        self.assertEqual(result["holdFrames"], 4)

    def test_reject_invalid_settings(self):
        with self.assertRaises(ValueError):
            server.handle_action({
                "action": "settings",
                "threshold": 120,
                "holdFrames": 4,
            })

    def test_word_suggestion(self):
        result = server.handle_action({
            "action": "suggestion",
            "word": "HELLO",
        })
        self.assertEqual(result["text"], "HELLO ")

    def test_clear_text(self):
        server.handle_action({
            "action": "suggestion",
            "word": "HELLO",
        })
        result = server.handle_action({
            "action": "clear"
        })
        self.assertEqual(result["text"], "")

    def test_reject_unknown_action(self):
        with self.assertRaises(ValueError):
            server.handle_action({
                "action": "invalid_action"
            })


if __name__ == "__main__":
    unittest.main()
