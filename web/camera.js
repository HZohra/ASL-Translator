
"use strict";

// SignSight camera module
// Uses the browser webcam API.
// No OpenCV or machine learning required.

const video = document.getElementById("cameraVideo");
const placeholder = document.getElementById("placeholder");
const badge = document.getElementById("cameraBadge");
const status = document.getElementById("cameraStatus");
const info = document.getElementById("cameraInfo");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const mirrorToggle = document.getElementById("mirrorToggle");

const resolutionLabel = document.getElementById("resolution");
const deviceStatus = document.getElementById("deviceStatus");

let stream = null;
let starting = false;

function updateStatus(message, isLive = false) {
  status.textContent = message;
  badge.textContent = isLive ? "● Live" : "Stopped";
  badge.classList.toggle("live", isLive);
}

async function startCamera() {
  if (starting || stream) return;

  starting = true;
  startButton.disabled = true;
  updateStatus("Requesting camera permission...");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires localhost or a secure HTTPS page."
      );
    }

    const newStream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      });

    stream = newStream;
    video.srcObject = stream;

    await video.play();

    placeholder.hidden = true;
    info.hidden = false;

    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();

    const width = video.videoWidth || settings.width || 0;
    const height = video.videoHeight || settings.height || 0;

    info.textContent =
      `${width} × ${height} webcam`;

    resolutionLabel.textContent =
      `Resolution: ${width} × ${height}`;

    deviceStatus.textContent =
      `Device: ${track.label || "Default webcam"}`;

    startButton.disabled = true;
    stopButton.disabled = false;

    updateStatus("Camera running — live video preview", true);

    track.addEventListener("ended", () => {
      stopCamera();
    });

  } catch (error) {
    stopCamera();

    let message = error.message;

    if (error.name === "NotAllowedError") {
      message = "Camera permission denied. Allow camera access.";
    } else if (error.name === "NotFoundError") {
      message = "No webcam was found.";
    } else if (error.name === "NotReadableError") {
      message = "Camera is busy or unavailable.";
    }

    status.textContent = `Camera error: ${message}`;
  } finally {
    starting = false;

    if (!stream) {
      startButton.disabled = false;
    }
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  video.pause();
  video.srcObject = null;

  placeholder.hidden = false;
  info.hidden = true;

  startButton.disabled = false;
  stopButton.disabled = true;

  resolutionLabel.textContent = "Resolution: —";
  deviceStatus.textContent = "Device: Not connected";

  updateStatus("Camera stopped.");
}

function toggleMirror() {
  video.classList.toggle(
    "mirrored",
    mirrorToggle.checked
  );
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
mirrorToggle.addEventListener("change", toggleMirror);

// Release the webcam when leaving the page.
window.addEventListener("pagehide", stopCamera);

// Ensure the hidden placeholder stays hidden while streaming.
const extraStyle = document.createElement("style");
extraStyle.textContent = `
  .placeholder[hidden],
  .camera-info[hidden] {
    display: none !important;
  }
`;
document.head.appendChild(extraStyle);
