"""
Flask web app that streams webcam video to the browser.
Applies Gaussian blur to the ENTIRE frame when a peace sign gesture is detected.
Uses smooth lerp transitions between clear and blurred states.
"""

import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template

from gesture_detector import GestureDetector
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

app = Flask(__name__)

# ---- Camera state ----
_camera: cv2.VideoCapture | None = None
_camera_lock = threading.Lock()

# ---- Blur state ----
_blur_amount: float = 0.0  # 0.0 = clear, 1.0 = fully blurred
_blur_lock = threading.Lock()

# ---- Configuration ----
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
BLUR_MAX_KERNEL = 51  # must be odd; max Gaussian blur kernel size
BLUR_LERP_SPEED = 0.12  # smoothing factor per frame (~0.3s at 15fps)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _get_camera() -> cv2.VideoCapture:
    """Lazily initialize and return the camera.

    Raises RuntimeError if the camera cannot be opened.
    """
    global _camera
    if _camera is None:
        with _camera_lock:
            if _camera is None:
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    cam.release()
                    raise RuntimeError(
                        "Cannot open webcam (index 0). "
                        "Check that a camera is connected and not in use by another app."
                    )
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cam.set(cv2.CAP_PROP_FPS, 30)
                _camera = cam
    return _camera


def _release_camera():
    """Release the camera resource."""
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None


def _draw_status_overlay(frame, is_blurred: bool, blur_level: float):
    """Draw a status pill in the top-left corner of the frame."""
    h, w = frame.shape[:2]

    if blur_level < 0.05:
        text = "Clear"
        color = (0, 255, 100)  # green
    elif blur_level > 0.95:
        text = "Blurred"
        color = (0, 100, 255)  # red-ish (BGR: blue + red = orange/amber)
    else:
        text = "Blurring..."
        color = (0, 200, 255)  # yellow

    # Background pill
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    pad_x, pad_y = 16, 10
    x, y = 20, 40
    cv2.rectangle(
        frame,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + pad_y),
        (0, 0, 0, 0),
        -1,
    )
    # Slight transparency for the pill background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + pad_y),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Text
    cv2.putText(frame, text, (x, y), FONT, 0.8, color, 2, cv2.LINE_AA)


def _draw_landmarks(frame, all_hand_landmarks, detector, h: int, w: int):
    """Draw hand skeleton with dots on finger joints.

    - Connections: thin white lines
    - Fingertips: larger circle — green if extended, red if folded
    - Joints: small white dots
    """
    connections = HandLandmarksConnections.HAND_CONNECTIONS

    for hand_lm in all_hand_landmarks:
        # Get pixel coordinates for all 21 landmarks
        px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

        # Finger states for color coding
        states = detector.finger_states(hand_lm)

        # Which landmarks belong to which finger
        finger_tip_indices = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }
        # Map tip index → finger name
        tip_to_name = {v: k for k, v in finger_tip_indices.items()}

        # Draw connections
        for conn in connections:
            p1 = px[conn.start]
            p2 = px[conn.end]
            cv2.line(frame, p1, p2, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw landmarks
        for i, (x, y) in enumerate(px):
            is_tip = i in tip_to_name
            finger_name = tip_to_name.get(i)

            if is_tip and finger_name:
                # Fingertip — larger, colored by extension state
                extended = states.get(finger_name, False)
                color = (0, 255, 0) if extended else (0, 80, 255)  # green : orange
                radius = 8
                # Glow ring
                cv2.circle(frame, (x, y), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            elif i == 0:
                # Wrist — neutral
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Joint — small dot
                cv2.circle(frame, (x, y), 3, (220, 220, 220), -1, cv2.LINE_AA)


def _error_frame(message: str) -> bytes:
    """Generate a single JPEG frame showing an error message."""
    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Camera Error",
        (CAMERA_WIDTH // 2 - 150, CAMERA_HEIGHT // 2 - 20),
        FONT,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (CAMERA_WIDTH // 2 - 280, CAMERA_HEIGHT // 2 + 40),
        FONT,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def generate_frames():
    """Generator that yields MJPEG frames with gesture-controlled blur."""
    global _blur_amount

    detector = None
    camera = None

    try:
        camera = _get_camera()
    except RuntimeError as e:
        # Camera unavailable — yield one error frame and exit
        yield _error_frame(str(e))
        return

    detector = GestureDetector()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Mirror the frame (like a selfie camera)
            frame = cv2.flip(frame, 1)

            # Detect peace sign gesture
            peace_detected, all_hands = detector.detect(frame)

            # Smooth lerp transition
            target = 1.0 if peace_detected else 0.0
            with _blur_lock:
                _blur_amount += (target - _blur_amount) * BLUR_LERP_SPEED
                blur = _blur_amount

            # Apply Gaussian blur when needed
            if blur > 0.005:
                # Kernel size: odd number scaling from 3 to BLUR_MAX_KERNEL
                kernel = int(3 + blur * (BLUR_MAX_KERNEL - 3)) | 1
                blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
                # Blend between clear and blurred
                if blur < 1.0:
                    frame = cv2.addWeighted(frame, 1.0 - blur, blurred, blur, 0)
                else:
                    frame = blurred

            # Draw hand tracking landmarks (after blur so they stay sharp)
            _draw_landmarks(
                frame, all_hands, detector, CAMERA_HEIGHT, CAMERA_WIDTH
            )

            # Draw status overlay
            _draw_status_overlay(frame, peace_detected, blur)

            # Encode as JPEG for MJPEG stream
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        if detector is not None:
            detector.close()


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Serve the developer about page."""
    return render_template("about.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  Gesture-Controlled Blur Cam")
    print("  Open http://localhost:5000 in your browser")
    print("  Make a peace sign to blur the camera!")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _release_camera()
