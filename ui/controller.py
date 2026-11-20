"""Prediction-to-text logic, independent of UI and JAX."""

class TranslationController:
    def __init__(self, threshold=0.80, frames_to_hold=8):
        self.threshold = threshold
        self.frames_to_hold = frames_to_hold
        self.text = ""
        self.held = 0
        self.current_label = None
        self.active_latch = None
        self.no_hand_frames = 0
        self.running = False
        self.last_added = None

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.held = 0
        self.current_label = None

    def reset_hold(self):
        self.held = 0
        self.current_label = None

    def process(self, prediction):
        """Process one prediction. A sign can be committed once per continuous hold.

        Returning to no-hand for 8 frames adds one space and re-arms the latch.
        This is a prototype policy that can be refined after model testing.
        """
        if not self.running:
            return
        if not prediction.get("hand_visible", False):
            self.held = 0
            self.current_label = None
            self.no_hand_frames += 1
            if self.no_hand_frames >= self.frames_to_hold and self.active_latch is not None:
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                    self.last_added = "space"
                self.active_latch = None
            return
        self.no_hand_frames = 0
        label = prediction.get("label")
        conf = prediction.get("confidence", 0)
        if not isinstance(label, str) or len(label) != 1 or not label.isalpha() or conf < self.threshold:
            self.reset_hold()
            return
        label = label.upper()
        if label != self.current_label:
            self.current_label = label
            self.held = 0
        if self.active_latch == label:
            return
        self.held += 1
        if self.held >= self.frames_to_hold:
            self.text += label
            self.last_added = label
            self.active_latch = label
            self.held = 0

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def delete(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self.last_added = None
        self.active_latch = None
        self.reset_hold()

    def settings(self, threshold_percent, frames_to_hold):
        self.threshold = threshold_percent / 100
        self.frames_to_hold = frames_to_hold
        self.reset_hold()
