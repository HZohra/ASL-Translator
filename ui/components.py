
"""SignSight dashboard for Jupyter.

This is a UI prototype using simulated predictions.
The real camera and JAX model are not connected yet.
"""

import html
import ipywidgets as widgets
from IPython.display import HTML, display

from .controller import TranslationController
from .mock_predictor import ALPHABET, SEQUENCE, mock_prediction


STYLE = """
<style>
.ss-app {
    background:#0b0e14;
    color:#e8ecf2;
    border:1px solid #252d3a;
    border-radius:16px;
    padding:20px;
    font-family:Inter,system-ui,sans-serif;
    max-width:1300px;
    box-sizing:border-box;
}
.ss-app * {box-sizing:border-box}
.ss-top {
    display:flex;align-items:center;
    justify-content:space-between;
    gap:14px;flex-wrap:wrap;margin-bottom:16px;
}
.ss-brand {display:flex;align-items:center;gap:12px}
.ss-brand-icon {
    background:#2563eb;padding:11px;
    border-radius:10px;font-size:19px;
}
.ss-name {font-weight:750;font-size:22px;color:#fff}
.ss-subtitle {
    color:#a3adbd;font-size:11px;margin-top:4px;
}
.ss-pills {display:flex;flex-wrap:wrap;gap:7px}
.ss-pill {
    border-radius:20px;border:1px solid #252d3a;
    background:#131821;padding:7px 10px;
    color:#bfc9d7;font-size:11px;
}
.ss-live {
    background:#10261b;
    border-color:#214e36;color:#7eedb5;
}
.ss-section {
    background:#131821;border:1px solid #252d3a;
    padding:15px;border-radius:14px;
    margin-bottom:12px;color:#e8ecf2;
}
.ss-heading {
    color:#a3adbd;letter-spacing:.09em;
    font-size:11px;font-weight:750;
    text-transform:uppercase;margin-bottom:12px;
}
.ss-flex {
    display:flex;justify-content:space-between;
    align-items:center;gap:8px;
}
.ss-muted {font-size:11px;color:#9aa6b7}
.ss-camera {
    height:385px;position:relative;background:#05070b;
    border:1px solid #252d3a;border-radius:13px;
    overflow:hidden;display:flex;align-items:center;
    justify-content:center;
}
.ss-camera-tag {
    position:absolute;top:12px;left:12px;
    border-radius:6px;background:#111722;
    padding:7px 9px;color:#fff;font-size:11px;
}
.ss-camera-empty {
    text-align:center;color:#8c9cb1;font-size:13px;
}
.ss-camera-empty div {
    font-size:36px;margin-bottom:9px;
}
.ss-camera-alert {
    position:absolute;bottom:12px;left:12px;right:12px;
    background:#0c121a;border:1px solid #263142;
    border-radius:9px;padding:11px;
    color:#cbd5e1;font-size:12px;
}
.ss-dot {
    width:8px;height:8px;display:inline-block;
    border-radius:50%;background:#718096;
    margin-right:8px;
}
.ss-dot.green {background:#3dd68c}
.ss-dot.amber {background:#f5b544}
.ss-sign-row {
    display:flex;align-items:center;gap:14px;
}
.ss-bigletter {
    background:#0b1a33;border:1px solid #1f3a66;
    border-radius:13px;width:96px;min-width:96px;
    height:96px;display:flex;align-items:center;
    justify-content:center;font-size:63px;
    font-weight:750;color:#e8f0ff;
}
.ss-metrics {min-width:0;flex:1}
.ss-percent {
    font-size:25px;font-weight:750;color:#5b9bff;
}
.ss-track {
    height:9px;background:#252d3a;
    border-radius:20px;overflow:hidden;
    margin:8px 0;position:relative;
}
.ss-track.small {height:7px;margin:0}
.ss-fill {
    height:100%;background:#5b9bff;border-radius:20px;
}
.ss-amber {background:#f5b544}
.ss-threshold {
    position:absolute;top:0;bottom:0;
    width:2px;background:#e8ecf2;
}
.ss-holdbar {display:flex;gap:4px;margin-top:9px}
.ss-segment {
    height:9px;flex:1;
    background:#252d3a;border-radius:3px;
}
.ss-segment.active {background:#3dd68c}
.ss-guess {
    display:grid;
    grid-template-columns:24px minmax(0,1fr) 38px;
    gap:9px;align-items:center;
    margin:12px 0;font-size:12px;
}
.ss-result {
    background:#090d13;border:1px solid #2e3746;
    border-radius:10px;min-height:61px;
    padding:14px;font-weight:700;
    font-size:26px;letter-spacing:.05em;
    color:#fff;white-space:pre-wrap;
    overflow-wrap:anywhere;
}
.ss-keys {display:flex;flex-wrap:wrap;gap:5px}
.ss-key {
    border:1px solid #2b3441;color:#a3adbd;
    background:#151c28;min-width:25px;
    text-align:center;border-radius:6px;
    padding:9px 5px;font-size:11px;
}
.ss-key.active {
    background:#2563eb;
    border-color:#2563eb;color:white;
}
.ss-app .widget-button {
    border-radius:9px;min-height:39px;
    font-weight:600;font-size:12px;
}
.ss-app .widget-button.mod-primary {
    background:#2563eb;color:#fff;border:0;
}
.ss-app .widget-button.mod-warning {
    background:#241a10;color:#f5c98a;
    border:1px solid #5a3a1e;
}
.ss-app .widget-checkbox label,
.ss-app .widget-label,
.ss-app .widget-readout {
    color:#d0d8e4!important;
    font-size:12px!important;
}
.ss-app .widget-int-slider {
    width:100%!important;
}


/* Make checkbox labels white */
.ss-app .widget-checkbox .widget-label,
.ss-app .widget-checkbox label,
.ss-app .widget-checkbox .widget-label span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Make slider numbers white */
.ss-app .widget-int-slider .widget-readout,
.ss-app .widget-slider .widget-readout,
.ss-app .jupyter-widgets-readout {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Force inherited text color on widget controls */
.ss-app .widget-checkbox,
.ss-app .widget-int-slider,
.ss-app .widget-slider {
    color: #ffffff !important;
    --jp-ui-font-color1: #ffffff;
    --jp-content-font-color1: #ffffff;
}


</style>
"""


class SignSightDemo:
    """Dashboard compatible with TranslationController."""

    def __init__(self):
        self.logic = TranslationController()
        self.demo_index = 0
        self.prediction = self._empty_prediction()

        # HTML dashboard components
        self.header = widgets.HTML()
        self.camera = widgets.HTML()
        self.current = widgets.HTML()
        self.guesses = widgets.HTML()
        self.translation = widgets.HTML()
        self.alphabet = widgets.HTML()
        self.notice = widgets.HTML()

        # Camera/demo controls
        self.start_btn = widgets.Button(
            description="Start Demo",
            icon="play",
            button_style="primary"
        )

        self.stop_btn = widgets.Button(
            description="Stop",
            icon="stop"
        )

        self.next_btn = widgets.Button(
            description="Next mock frame",
            icon="step-forward"
        )

        # Translation controls
        self.space_btn = widgets.Button(description="Space")

        self.delete_btn = widgets.Button(
            description="Delete",
            icon="backward"
        )

        self.clear_btn = widgets.Button(
            description="Clear",
            button_style="warning"
        )

        self.copy_btn = widgets.Button(
            description="Copy text",
            icon="copy"
        )

        # Native ipywidgets labels/readouts can render black in VS Code.
        # Hide them and render white labels as HTML widgets instead.
        self.mirror = widgets.Checkbox(
            value=True,
            description="",
            indent=False,
            layout=widgets.Layout(width="26px")
        )
        self.skeleton = widgets.Checkbox(
            value=True,
            description="",
            indent=False,
            layout=widgets.Layout(width="26px")
        )
        self.mirror_label = widgets.HTML(
            '<span style="color:#ffffff!important;font-size:13px">Mirror view</span>'
        )
        self.skeleton_label = widgets.HTML(
            '<span style="color:#ffffff!important;font-size:13px">Hand skeleton</span>'
        )

        self.threshold = widgets.IntSlider(
            value=80, min=50, max=95, step=5,
            description="", readout=False,
            layout=widgets.Layout(width="100%")
        )
        self.hold = widgets.IntSlider(
            value=8, min=4, max=16, step=1,
            description="", readout=False,
            layout=widgets.Layout(width="100%")
        )
        self.threshold_label = widgets.HTML(
            '<span style="color:#ffffff!important;font-size:13px">Confidence</span>'
        )
        self.hold_label = widgets.HTML(
            '<span style="color:#ffffff!important;font-size:13px">Hold frames</span>'
        )
        self.threshold_value = widgets.HTML()
        self.hold_value = widgets.HTML()
        self._refresh_slider_values()

        # Suggested words
        self.suggestions = [
            widgets.Button(
                description="",
                layout=widgets.Layout(width="auto")
            )
            for _ in range(3)
        ]

        # Connect buttons to methods
        self.start_btn.on_click(self.start)
        self.stop_btn.on_click(self.stop)
        self.next_btn.on_click(self.next_frame)

        self.space_btn.on_click(
            lambda _: self.edit("space")
        )
        self.delete_btn.on_click(
            lambda _: self.edit("delete")
        )
        self.clear_btn.on_click(
            lambda _: self.edit("clear")
        )

        self.copy_btn.on_click(self.copy_text)

        # React to settings changes
        self.threshold.observe(
            self.update_settings,
            names="value"
        )
        self.hold.observe(
            self.update_settings,
            names="value"
        )

        self.mirror.observe(
            lambda _: self.render(),
            names="value"
        )
        self.skeleton.observe(
            lambda _: self.render(),
            names="value"
        )

        for i, button in enumerate(self.suggestions):
            button.on_click(
                lambda _, idx=i: self.use_suggestion(idx)
            )

        self.render()

    @staticmethod
    def _empty_prediction():
        return {
            "hand_visible": False,
            "label": None,
            "confidence": 0.0,
            "top3": []
        }

    # -----------------------------
    # Start and Stop
    # -----------------------------

    def start(self, _=None):
        self.logic.start()
        self.render()

    def stop(self, _=None):
        self.logic.stop()
        self.prediction = self._empty_prediction()
        self.render()

    # -----------------------------
    # Mock prediction simulation
    # -----------------------------

    def next_frame(self, _=None):
        if not self.logic.running:
            return

        self.prediction = mock_prediction(
            self.demo_index
        )

        self.logic.process(self.prediction)

        if not self.prediction["hand_visible"]:
            if (
                self.logic.no_hand_frames
                >= self.logic.frames_to_hold
            ):
                self.demo_index = (
                    self.demo_index + 1
                ) % len(SEQUENCE)

        elif (
            self.logic.active_latch
            == self.prediction["label"]
            and self.logic.held == 0
        ):
            old_label = self.prediction["label"]

            self.demo_index = (
                self.demo_index + 1
            ) % len(SEQUENCE)

            # Simulated release for repeated letters
            if SEQUENCE[self.demo_index] == old_label:
                self.logic.active_latch = None

        self.render()

    # -----------------------------
    # Settings
    # -----------------------------

    def _refresh_slider_values(self):
        """White HTML readouts bypass VS Code's native widget theme."""
        self.threshold_value.value = (
            f'<span style="color:#ffffff!important;font-weight:700;font-size:13px">'
            f'{self.threshold.value}%</span>'
        )
        self.hold_value.value = (
            f'<span style="color:#ffffff!important;font-weight:700;font-size:13px">'
            f'{self.hold.value}</span>'
        )

    def update_settings(self, _=None):
        self.logic.settings(
            self.threshold.value,
            self.hold.value
        )
        self._refresh_slider_values()
        self.render()

    # -----------------------------
    # Text editing
    # -----------------------------

    def edit(self, action):
        getattr(self.logic, action)()
        self.render()

    # -----------------------------
    # Word suggestions
    # -----------------------------

    def _suggestions(self):
        words = (
            "HELLO", "HELP", "HOW",
            "WORLD", "WORK", "WATER",
            "WHAT", "WHERE", "THANK",
            "THANKS", "THE", "YOU",
            "YES", "YOUR", "PLEASE",
            "SORRY", "GOOD", "MORNING"
        )

        sentence = self.logic.text

        if not sentence or sentence.endswith(" "):
            return ["HELLO", "THANK", "PLEASE"]

        fragment = sentence.split()[-1].upper()

        return [
            word for word in words
            if word.startswith(fragment)
            and word != fragment
        ][:3]

    def use_suggestion(self, idx):
        suggestions = self._suggestions()

        if idx >= len(suggestions):
            return

        current = self.logic.text

        if current and not current.endswith(" "):
            fragment = current.split()[-1]
            prefix = current[:-len(fragment)]
        else:
            prefix = current

        self.logic.text = (
            prefix + suggestions[idx] + " "
        )

        self.logic.active_latch = None
        self.logic.reset_hold()
        self.render()

    # -----------------------------
    # Copy text
    # -----------------------------

    def copy_text(self, _=None):
        # Fallback for notebook environments:
        # display selectable text for manual copying.
        if hasattr(self, "copy_area"):
            self.copy_area.value = self.logic.text
        else:
            self.copy_area = widgets.Textarea(
                value=self.logic.text,
                layout=widgets.Layout(
                    width="100%",
                    height="70px"
                )
            )
            if not hasattr(self, "copy_slot"):
                self.copy_slot = widgets.VBox([])
            self.copy_slot.children = (self.copy_area,)

        self.notice.value = (
            '<span class="ss-muted">'
            'Select the text below and press Ctrl+C.'
            '</span>'
        )

    # -----------------------------
    # Update dashboard
    # -----------------------------

    def render(self):
        p = self.prediction
        running = self.logic.running

        visible = bool(
            running and p.get("hand_visible")
        )

        label = (
            str(p.get("label") or "—")
            if visible else "—"
        )

        confidence = (
            max(
                0,
                min(
                    100,
                    round(
                        float(
                            p.get("confidence", 0)
                        ) * 100
                    )
                )
            )
            if visible else 0
        )

        low = (
            visible
            and confidence < self.threshold.value
        )

        accent = (
            "#f5b544" if low else "#5b9bff"
        )

        # Header
        pill = (
            '<span class="ss-pill ss-live">'
            '● Demo running</span>'
            if running else
            '<span class="ss-pill">● Stopped</span>'
        )

        self.header.value = (
            '<div class="ss-top">'
            '<div class="ss-brand">'
            '<span class="ss-brand-icon">✋</span>'
            '<div>'
            '<div class="ss-name">'
            'SignSight '
            '<span style="color:#a3adbd">'
            '· ASL → Text</span></div>'
            '<div class="ss-subtitle">'
            'Sign a letter and hold steady. '
            'Predictions are simulated for now.'
            '</div></div></div>'
            f'<div class="ss-pills">{pill}'
            '<span class="ss-pill">FPS: —</span>'
            '<span class="ss-pill">Latency: —</span>'
            '<span class="ss-pill">'
            'JAX: not connected</span>'
            '</div></div>'
        )

        # Camera status
        if not running:
            msg = "Demo stopped"
            detail = "Press Start Demo to begin."
            dot = ""

        elif not visible:
            msg = "No hand in view"
            detail = "Click Next mock frame to advance."
            dot = ""

        elif low:
            msg = "Low confidence"
            detail = (
                "The prediction is below your threshold."
            )
            dot = "amber"

        else:
            msg = (
                f'Reading “{html.escape(label)}”'
            )
            detail = "Hold steady to add a letter."
            dot = "green"

        mirror_label = (
            "Mirrored"
            if self.mirror.value else "Normal"
        )

        self.camera.value = (
            '<div class="ss-camera">'
            '<div class="ss-camera-tag">'
            '640 × 480 placeholder &nbsp; '
            f'{mirror_label}</div>'
            '<div class="ss-camera-empty">'
            '<div>▣</div>'
            'Webcam will appear here'
            '<br>'
            '<small>'
            'Demo mode — no camera connected'
            '</small></div>'
            '<div class="ss-camera-alert">'
            f'<span class="ss-dot {dot}"></span>'
            f'<b>{msg}</b> '
            '<span style="color:#a3adbd">'
            f'{detail}</span>'
            '</div></div>'
        )

        # Current sign
        held = min(
            self.logic.held,
            self.logic.frames_to_hold
        )

        segments = "".join(
            '<i class="ss-segment%s"></i>'
            % (" active" if i < held else "")
            for i in range(
                self.logic.frames_to_hold
            )
        )

        self.current.value = (
            '<div class="ss-section">'
            '<div class="ss-flex">'
            '<div class="ss-heading">'
            'Current sign</div>'
            '<span class="ss-muted">'
            'Mock prediction</span></div>'
            '<div class="ss-sign-row">'
            '<div class="ss-bigletter">'
            f'{html.escape(label)}</div>'
            '<div class="ss-metrics">'
            '<div class="ss-flex">'
            '<span class="ss-muted">'
            'Confidence</span>'
            '<span class="ss-percent" '
            f'style="color:{accent}">'
            f'{confidence}%</span></div>'
            '<div class="ss-track">'
            '<div class="ss-fill '
            f'{"ss-amber" if low else ""}" '
            f'style="width:{confidence}%">'
            '</div>'
            '<div class="ss-threshold" '
            f'style="left:{self.threshold.value}%">'
            '</div></div>'
            '<div class="ss-muted">'
            'White line = minimum confidence '
            f'({self.threshold.value}%)'
            '</div></div></div>'
            '<div class="ss-flex" '
            'style="margin-top:16px">'
            '<span class="ss-muted">'
            'Hold steady to add</span>'
            '<span class="ss-muted">'
            f'{held} / '
            f'{self.logic.frames_to_hold} frames'
            '</span></div>'
            '<div class="ss-holdbar">'
            f'{segments}</div></div>'
        )

        # Top three guesses
        guess_rows = ""

        entries = (
            p.get("top3", [])
            if visible else []
        )

        for guess, prob in entries[:3]:
            pct = max(
                0,
                min(100, round(float(prob) * 100))
            )

            guess_rows += (
                '<div class="ss-guess">'
                f'<b>{html.escape(str(guess))}</b>'
                '<div class="ss-track small">'
                '<div class="ss-fill" '
                f'style="width:{pct}%">'
                '</div></div>'
                f'<span>{pct}%</span>'
                '</div>'
            )

        if not guess_rows:
            guess_rows = (
                '<div class="ss-muted">'
                'No prediction yet</div>'
            )

        self.guesses.value = (
            '<div class="ss-section">'
            '<div class="ss-heading">'
            'Top 3 guesses</div>'
            f'{guess_rows}</div>'
        )

        # Translated text
        sentence = self.logic.text

        safe_text = (
            html.escape(sentence)
            if sentence else "—"
        )

        self.translation.value = (
            '<div class="ss-section">'
            '<div class="ss-flex">'
            '<div class="ss-heading">'
            'Translated text</div>'
            '<span class="ss-muted">'
            f'{len(sentence)} characters · '
            f'{len(sentence.split())} words'
            '</span></div>'
            '<div class="ss-result">'
            f'{safe_text}</div></div>'
        )

        # Alphabet
        keys = "".join(
            '<span class="ss-key '
            f'{"active" if visible and c == label else ""}'
            f'">{c}</span>'
            for c in ALPHABET
        )

        self.alphabet.value = (
            '<div class="ss-heading">'
            'Alphabet · current sign highlighted '
            '(J and Z require motion)'
            '</div>'
            f'<div class="ss-keys">{keys}</div>'
        )

        # Suggestions
        suggestions = self._suggestions()

        for idx, button in enumerate(
            self.suggestions
        ):
            if idx < len(suggestions):
                button.description = suggestions[idx]
                button.layout.display = ""
            else:
                button.description = ""
                button.layout.display = "none"

    # -----------------------------
    # Display Jupyter dashboard
    # -----------------------------

    def show(self):
        display(HTML(STYLE))

        left = widgets.VBox(
            [
                self.camera,
                widgets.HBox(
                    [
                        self.start_btn,
                        self.stop_btn,
                        self.next_btn
                    ],
                    layout=widgets.Layout(
                        flex_flow="row wrap",
                        gap="8px"
                    )
                ),
                widgets.HBox(
                    [
                        widgets.HBox(
                            [self.mirror, self.mirror_label],
                            layout=widgets.Layout(align_items="center", gap="5px")
                        ),
                        widgets.HBox(
                            [self.skeleton, self.skeleton_label],
                            layout=widgets.Layout(align_items="center", gap="5px")
                        )
                    ],
                    layout=widgets.Layout(
                        align_items="center", flex_flow="row wrap", gap="22px"
                    )
                )
            ],
            layout=widgets.Layout(
                width="60%",
                min_width="0"
            )
        )

        settings = widgets.VBox(
            [
                widgets.HTML('<div class="ss-heading">Settings</div>'),
                widgets.HBox(
                    [
                        widgets.Box([self.threshold_label], layout=widgets.Layout(width="105px")),
                        self.threshold,
                        widgets.Box([self.threshold_value], layout=widgets.Layout(width="50px"))
                    ],
                    layout=widgets.Layout(align_items="center", gap="9px", width="100%")
                ),
                widgets.HBox(
                    [
                        widgets.Box([self.hold_label], layout=widgets.Layout(width="105px")),
                        self.hold,
                        widgets.Box([self.hold_value], layout=widgets.Layout(width="50px"))
                    ],
                    layout=widgets.Layout(align_items="center", gap="9px", width="100%")
                )
            ],
            layout=widgets.Layout(padding="4px 2px", width="100%")
        )

        right = widgets.VBox(
            [
                self.current,
                self.guesses,
                settings
            ],
            layout=widgets.Layout(
                width="38%",
                min_width="0"
            )
        )

        columns = widgets.HBox(
            [left, right],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="flex-start",
                gap="14px"
            )
        )

        editing = widgets.HBox(
            [
                self.space_btn,
                self.delete_btn,
                self.clear_btn
            ],
            layout=widgets.Layout(
                flex_flow="row wrap",
                gap="8px"
            )
        )

        completions = widgets.HBox(
            [
                widgets.HTML(
                    '<span class="ss-muted">'
                    'Finish word:</span>'
                ),
                *self.suggestions
            ],
            layout=widgets.Layout(
                flex_flow="row wrap",
                align_items="center",
                gap="8px"
            )
        )

        self.copy_slot = widgets.VBox([])

        body = widgets.VBox(
            [
                self.header,
                columns,
                self.translation,
                editing,
                completions,
                self.copy_btn,
                self.notice,
                self.copy_slot,
                self.alphabet
            ],
            layout=widgets.Layout(
                width="100%"
            )
        )

        body.add_class("ss-app")
        display(body)
