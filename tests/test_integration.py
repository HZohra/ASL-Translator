
"""SignSight integration tests.

Tests:
Model Adapter -> HTTP API -> TranslationController

No OpenCV or trained model required.
"""

import json
import threading
import unittest

from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from backend import server
from ui.controller import TranslationController
from ui.model_adapter import to_prediction


class SignSightIntegrationTests(unittest.TestCase):

    def setUp(self):
        """Create a fresh controller and HTTP server for every test."""

        # Reset Python translation state.
        server.controller = TranslationController()
        server.demo_index = 0

        server.prediction = {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": [],
        }

        # Start a temporary local HTTP server.
        # Port 0 tells Python to choose an available port.
        self.http_server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            server.SignSightHandler,
        )

        self.port = self.http_server.server_address[1]

        self.thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True,
        )

        self.thread.start()

    def tearDown(self):
        """Stop the temporary HTTP server after each test."""

        self.http_server.shutdown()
        self.http_server.server_close()

        self.thread.join(timeout=5)

    def post(self, action, **values):
        """Send an actual HTTP POST request to the Python API."""

        body = json.dumps({
            "action": action,
            **values,
        }).encode("utf-8")

        request = Request(
            f"http://127.0.0.1:{self.port}/api/action",
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def get_state(self):
        """Retrieve the current application state using HTTP GET."""

        url = f"http://127.0.0.1:{self.port}/api/state"

        with urlopen(url, timeout=5) as response:
            return json.load(response)

    def make_prediction(self, letter="A", confidence=0.92):
        """Simulate neural-network output probabilities."""

        remaining = 1.0 - confidence

        return to_prediction(
            [
                confidence,
                remaining * 0.75,
                remaining * 0.25,
            ],
            [letter, "B", "C"],
        )

    # --------------------------------------------------
    # TEST 1: API CONNECTION
    # --------------------------------------------------

    def test_api_connection(self):
        """The API returns a valid initial application state."""

        state = self.get_state()

        self.assertFalse(state["running"])
        self.assertEqual(state["text"], "")
        self.assertFalse(state["modelConnected"])

    # --------------------------------------------------
    # TEST 2: PREDICTION REACHES API STATE
    # --------------------------------------------------

    def test_prediction_reaches_dashboard_state(self):
        """Predictions reach the state consumed by the dashboard."""

        self.post("start")

        prediction = self.make_prediction(
            letter="A",
            confidence=0.92,
        )

        self.post(
            "prediction",
            prediction=prediction,
        )

        state = self.get_state()

        self.assertEqual(
            state["prediction"]["label"],
            "A",
        )

        self.assertAlmostEqual(
            state["prediction"]["confidence"],
            0.92,
        )

        self.assertEqual(
            state["prediction"]["top3"][0][0],
            "A",
        )

    # --------------------------------------------------
    # TEST 3: LETTER ACCEPTED AFTER HOLD
    # --------------------------------------------------

    def test_letter_accepted_after_hold(self):
        """Four confident predictions should add one letter."""

        self.post(
            "settings",
            threshold=80,
            holdFrames=4,
        )

        self.post("start")

        prediction = self.make_prediction(
            letter="A",
            confidence=0.92,
        )

        for _ in range(4):
            self.post(
                "prediction",
                prediction=prediction,
            )

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "A",
        )

    # --------------------------------------------------
    # TEST 4: LOW CONFIDENCE IS IGNORED
    # --------------------------------------------------

    def test_low_confidence_does_not_add_letter(self):
        """Predictions below the threshold must be rejected."""

        self.post(
            "settings",
            threshold=95,
            holdFrames=4,
        )

        self.post("start")

        prediction = self.make_prediction(
            letter="A",
            confidence=0.92,
        )

        for _ in range(8):
            self.post(
                "prediction",
                prediction=prediction,
            )

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "",
        )

    # --------------------------------------------------
    # TEST 5: NO DUPLICATE LETTERS
    # --------------------------------------------------

    def test_continuous_hold_does_not_duplicate(self):
        """Holding one sign should not add duplicate letters."""

        self.post(
            "settings",
            threshold=80,
            holdFrames=4,
        )

        self.post("start")

        prediction = self.make_prediction(
            letter="A",
            confidence=0.92,
        )

        for _ in range(12):
            self.post(
                "prediction",
                prediction=prediction,
            )

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "A",
        )

    # --------------------------------------------------
    # TEST 6: TEXT ACTIONS THROUGH HTTP
    # --------------------------------------------------

    def test_text_actions_work_through_http(self):
        """Browser-style actions update text through HTTP."""

        # Selecting HELLO automatically adds a space.
        self.post(
            "suggestion",
            word="HELLO",
        )

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "HELLO ",
        )

        # Pressing Space should not add duplicate spaces.
        self.post("space")

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "HELLO ",
        )

        # Delete removes the trailing space.
        self.post("delete")

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "HELLO",
        )

        # Clear removes all translated text.
        self.post("clear")

        state = self.get_state()

        self.assertEqual(
            state["text"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
