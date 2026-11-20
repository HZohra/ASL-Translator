
"""Local SignSight server — no OpenCV required."""

import json
import sys

from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from http.server import (
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.controller import TranslationController
from ui.mock_predictor import SEQUENCE, mock_prediction


lock = Lock()
controller = TranslationController()
demo_index = 0

prediction = {
    "hand_visible": False,
    "label": None,
    "confidence": 0.0,
    "top3": [],
}


def get_state():
    return {
        "running": controller.running,
        "text": controller.text,
        "held": controller.held,
        "holdFrames": controller.frames_to_hold,
        "threshold": round(controller.threshold * 100),
        "lastAdded": controller.last_added,
        "prediction": prediction if controller.running else {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": [],
        },
        "mode": "mock",
        "modelConnected": False,
    }


def handle_action(data):
    global demo_index, prediction

    action = data.get("action")

    if action == "start":
        controller.start()

    elif action == "stop":
        controller.stop()

    elif action == "mock":
        if controller.running:
            prediction = mock_prediction(demo_index)
            controller.process(prediction)

            if not prediction["hand_visible"]:
                if (
                    controller.no_hand_frames
                    >= controller.frames_to_hold
                ):
                    demo_index = (
                        demo_index + 1
                    ) % len(SEQUENCE)

            elif (
                controller.active_latch == prediction["label"]
                and controller.held == 0
            ):
                previous = prediction["label"]

                demo_index = (
                    demo_index + 1
                ) % len(SEQUENCE)

                if SEQUENCE[demo_index] == previous:
                    controller.active_latch = None
                    controller.reset_hold()

    elif action == "settings":
        threshold = int(data.get("threshold", 80))
        frames = int(data.get("holdFrames", 8))

        if not 50 <= threshold <= 95:
            raise ValueError("Invalid confidence threshold")

        if not 4 <= frames <= 16:
            raise ValueError("Invalid frame count")

        controller.settings(threshold, frames)

    elif action in ("space", "delete", "clear"):
        getattr(controller, action)()

    elif action == "suggestion":
        word = data.get("word", "")

        allowed = (
            "HELLO", "HELP", "HOW", "HI", "WORLD",
            "WORK", "WATER", "WHAT", "WHERE",
            "THANK", "THANKS", "THE", "YOU",
            "YES", "YOUR", "PLEASE", "SORRY",
            "GOOD", "MORNING",
        )

        if word not in allowed:
            raise ValueError("Unknown suggestion")

        original = controller.text

        if original.strip() and not original.endswith(" "):
            fragment = original.split()[-1]
            prefix = original[:-len(fragment)]
        else:
            prefix = original

        controller.text = prefix + word + " "
        controller.active_latch = None
        controller.reset_hold()

    elif action == "reset_demo":
        demo_index = 0
        controller.clear()
        controller.no_hand_frames = 0

        prediction = {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": [],
        }

    elif action == "prediction":
        incoming = data.get("prediction")

        if not isinstance(incoming, dict):
            raise ValueError("Invalid prediction")

        visible = incoming.get("hand_visible")

        if not isinstance(visible, bool):
            raise ValueError("Invalid hand visibility")

        if visible:
            label = incoming.get("label")
            confidence = incoming.get("confidence")

            if (
                not isinstance(label, str)
                or len(label) != 1
                or not label.isalpha()
            ):
                raise ValueError("Expected a single ASL letter")

            if (
                not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise ValueError("Invalid confidence")

        top3 = incoming.get("top3", [])

        if not isinstance(top3, list) or len(top3) > 3:
            raise ValueError("Invalid top 3 predictions")

        prediction = {
            "hand_visible": visible,
            "label": incoming.get("label"),
            "confidence": incoming.get("confidence", 0),
            "top3": top3,
        }

        controller.process(prediction)

    else:
        raise ValueError("Unknown action")

    return get_state()


class SignSightHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(ROOT),
            **kwargs,
        )

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == "/api/state":
            with lock:
                self.send_json(200, get_state())
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/action":
            self.send_json(
                404,
                {"error": "Unknown endpoint"},
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if not 0 < length <= 8192:
                raise ValueError("Invalid request size")

            data = json.loads(
                self.rfile.read(length)
            )

            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")

            with lock:
                result = handle_action(data)

            self.send_json(200, result)

        except (ValueError, TypeError) as error:
            self.send_json(
                400,
                {"error": str(error)},
            )


if __name__ == "__main__":
    print("SignSight server running")
    print("Open: http://127.0.0.1:8000/web/signsight.html")
    print("Stop the server with Ctrl+C")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 8000),
        SignSightHandler,
    )

    server.serve_forever()
