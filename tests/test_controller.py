import unittest
from ui.controller import TranslationController

class ControllerTests(unittest.TestCase):
    def test_accept_after_hold(self):
        c = TranslationController(frames_to_hold=3)
        c.start()
        for _ in range(3):
            c.process({"hand_visible": True, "label": "A", "confidence": .95})
        self.assertEqual(c.text, 'A')

    def test_no_duplicate_on_continuous_hold(self):
        c = TranslationController(frames_to_hold=2)
        c.start()
        for _ in range(12):
            c.process({"hand_visible": True, "label": "A", "confidence": .95})
        self.assertEqual(c.text, 'A')

    def test_low_confidence_is_ignored(self):
        c = TranslationController(frames_to_hold=2)
        c.start()
        for _ in range(5):
            c.process({"hand_visible": True, "label": "A", "confidence": .4})
        self.assertEqual(c.text, '')

    def test_clear_and_delete(self):
        c = TranslationController()
        c.text = 'HELLO'
        c.delete()
        self.assertEqual(c.text, 'HELL')
        c.clear()
        self.assertEqual(c.text, '')

if __name__ == '__main__':
    unittest.main()
