
"use strict";

// SignSight frontend-to-Python bridge.
// The browser handles display and the webcam.
// Python handles translation logic.

const $ = id => document.getElementById(id);

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const words = [
  "HELLO", "HELP", "HOW", "HI",
  "WORLD", "WORK", "WATER", "WHAT",
  "WHERE", "THANK", "THANKS", "THE",
  "YOU", "YES", "YOUR", "PLEASE",
  "SORRY", "GOOD", "MORNING"
];

let state = null;
let pending = false;


// Read the current state from Python.
async function getState() {
  const response = await fetch("/api/state", {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error("Python backend unavailable");
  }

  return response.json();
}


// Send commands to Python.
async function act(action, values = {}) {
  if (pending) return;

  pending = true;

  try {
    const response = await fetch("/api/action", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        action,
        ...values
      })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.error || "Request failed"
      );
    }

    state = result;
    render();

  } catch (error) {
    $("textStatus").textContent =
      "Python connection error: " + error.message;

  } finally {
    pending = false;
  }
}


// Update everything on the dashboard.
function render() {
  if (!state) return;

  const p = state.prediction || {};
  const visible = state.running && p.hand_visible;

  const letter = visible ? p.label : null;

  const confidence = visible
    ? Math.round(p.confidence * 100)
    : 0;

  $("currentLetter").textContent = letter || "—";
  $("confidence").textContent = confidence + "%";

  $("confidenceFill").style.width =
    confidence + "%";

  $("confidenceFill").style.background =
    confidence < state.threshold
      ? "#f5b544"
      : "#5b9bff";

  $("thresholdMark").style.left =
    state.threshold + "%";

  $("threshold").value = state.threshold;

  $("thresholdValue").textContent =
    state.threshold + "%";

  $("holdFrames").value = state.holdFrames;

  $("holdValue").textContent =
    state.holdFrames;

  $("holdCount").textContent =
    `${state.held} / ${state.holdFrames} frames`;

  // Hold-progress segments.
  const progress = $("holdSegments");
  progress.replaceChildren();

  for (let i = 0; i < state.holdFrames; i++) {
    const segment = document.createElement("div");

    segment.className =
      "segment" +
      (i < state.held ? " active" : "");

    progress.appendChild(segment);
  }

  // Top three model guesses.
  const top = $("topGuesses");
  top.replaceChildren();

  const guesses = visible
    ? (p.top3 || [])
    : [];

  if (!guesses.length) {
    top.textContent = "No prediction yet";
  }

  for (const [name, value] of guesses.slice(0, 3)) {
    const row = document.createElement("div");
    row.className = "guess";

    const label = document.createElement("strong");
    label.textContent = name;

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width =
      Math.round(value * 100) + "%";

    bar.append(fill);

    const percentage = document.createElement("span");
    percentage.textContent =
      Math.round(value * 100) + "%";

    row.append(label, bar, percentage);
    top.append(row);
  }

  // Translated text.
  $("translatedText").textContent = state.text;

  const count = state.text.trim()
    ? state.text.trim().split(/\s+/).length
    : 0;

  $("textCount").textContent =
    `${state.text.length} characters · ${count} words`;

  // Highlight current letter.
  document.querySelectorAll(".key").forEach(key => {
    key.classList.toggle(
      "active",
      key.textContent === letter
    );
  });

  $("demoFrame").disabled = !state.running;

  renderSuggestions();
}


// Word suggestions.
function renderSuggestions() {
  const container = $("suggestionButtons");
  container.replaceChildren();

  const text = state.text;
  let suggestions = [];

  if (!text || /\s$/.test(text)) {
    suggestions = ["HELLO", "THANK", "PLEASE"];

  } else {
    const fragment = text
      .trim()
      .split(/\s+/)
      .at(-1)
      .toUpperCase();

    suggestions = words.filter(word =>
      word.startsWith(fragment) &&
      word !== fragment
    ).slice(0, 3);
  }

  for (const word of suggestions) {
    const button = document.createElement("button");
    button.textContent = word;

    button.addEventListener("click", () => {
      act("suggestion", { word });
    });

    container.append(button);
  }
}


// Build the alphabet reference.
for (const letter of alphabet) {
  const key = document.createElement("div");

  key.className = "key";
  key.textContent = letter;

  $("alphabet").append(key);
}


// Mock prediction controls.
$("demoFrame").addEventListener(
  "click",
  () => act("mock")
);

$("resetDemo").addEventListener(
  "click",
  () => act("reset_demo")
);


// Text editing.
$("spaceBtn").addEventListener(
  "click",
  () => act("space")
);

$("deleteBtn").addEventListener(
  "click",
  () => act("delete")
);

$("clearBtn").addEventListener(
  "click",
  () => act("clear")
);


// Settings.
$("threshold").addEventListener("change", event => {
  act("settings", {
    threshold: Number(event.target.value),
    holdFrames: Number($("holdFrames").value)
  });
});

$("holdFrames").addEventListener("change", event => {
  act("settings", {
    threshold: Number($("threshold").value),
    holdFrames: Number(event.target.value)
  });
});


// Copy text.
$("copyBtn").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(
      state?.text || ""
    );

    $("textStatus").textContent = "Copied.";

  } catch {
    $("textStatus").textContent =
      "Copy unavailable. Select the text manually.";
  }
});


// Speak text.
$("speakBtn").addEventListener("click", () => {
  if (!state?.text?.trim()) {
    $("textStatus").textContent =
      "Nothing to speak.";
    return;
  }

  if (!("speechSynthesis" in window)) {
    $("textStatus").textContent =
      "Speech unavailable.";
    return;
  }

  speechSynthesis.cancel();

  speechSynthesis.speak(
    new SpeechSynthesisUtterance(state.text)
  );
});


// Translation controls.
// These are separate from webcam Start/Stop.
$("startTranslation").addEventListener(
  "click",
  () => act("start")
);

$("stopTranslation").addEventListener(
  "click",
  () => act("stop")
);



let polling = false;
let connectionLost = false;

// Read the latest Python state automatically.
async function refreshState() {
  // Avoid overlapping requests or overwriting
  // an action that is currently being processed.
  if (polling || pending) return;

  polling = true;

  try {
    const updatedState = await getState();

    // An action may have started while the GET was
    // in progress. Don't replace its newer state.
    if (!pending) {
      state = updatedState;
      render();
    }

    if (connectionLost) {
      $("textStatus").textContent =
        "Python backend reconnected.";
    }

    connectionLost = false;

  } catch (error) {
    if (!connectionLost) {
      $("textStatus").textContent =
        "Connection lost: " + error.message;
    }

    connectionLost = true;

  } finally {
    polling = false;
  }
}

// Initial connection.
refreshState();

// Refresh approximately 5 times per second.
const stateTimer = setInterval(refreshState, 200);

// Stop polling when leaving the page.
window.addEventListener("pagehide", () => {
  clearInterval(stateTimer);
});

