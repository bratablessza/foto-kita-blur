"""
Gesture detector using MediaPipe Hands (Tasks API).
Detects the "peace sign" gesture: index and middle fingers extended upward,
ring and pinky fingers folded down.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model(path: str | None = None) -> str:
    """Return the model path, downloading if necessary."""
    model_path = path or _DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Downloading hand landmarker model to {model_path}...")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path


class GestureDetector:
    """Detects peace sign hand gesture using MediaPipe Hands Tasks API.

    Landmark indices:
        4  = THUMB_TIP
        6  = INDEX_FINGER_PIP   →  8  = INDEX_FINGER_TIP
        10 = MIDDLE_FINGER_PIP  → 12 = MIDDLE_FINGER_TIP
        14 = RING_FINGER_PIP    → 16 = RING_FINGER_TIP
        18 = PINKY_PIP          → 20 = PINKY_TIP
    """

    def __init__(
        self,
        model_path: str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model = _ensure_model(model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    @staticmethod
    def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
        """A finger is extended if its tip has a smaller y than its PIP joint.

        In image coordinates y=0 is the top, so a pointing-up finger
        has tip.y < pip.y.
        """
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def is_peace_sign(self, hand_landmarks) -> bool:
        """Check if a single hand is making the peace sign gesture.

        Peace sign = index + middle fingers extended UP,
                     ring + pinky fingers folded DOWN.

        Args:
            hand_landmarks: List of 21 NormalizedLandmark objects.
        """
        return (
            self._is_finger_extended(hand_landmarks, 8, 6)       # index extended
            and self._is_finger_extended(hand_landmarks, 12, 10)  # middle extended
            and not self._is_finger_extended(hand_landmarks, 16, 14)  # ring folded
            and not self._is_finger_extended(hand_landmarks, 20, 18)  # pinky folded
        )

    def finger_states(self, hand_landmarks) -> dict[str, bool]:
        """Return which fingers are extended on a hand."""
        return {
            "thumb": self._is_finger_extended(hand_landmarks, 4, 3),
            "index": self._is_finger_extended(hand_landmarks, 8, 6),
            "middle": self._is_finger_extended(hand_landmarks, 12, 10),
            "ring": self._is_finger_extended(hand_landmarks, 16, 14),
            "pinky": self._is_finger_extended(hand_landmarks, 20, 18),
        }

    def detect(self, frame_bgr) -> tuple[bool, list]:
        """Process a BGR frame and detect peace sign.

        Args:
            frame_bgr: BGR image from OpenCV (numpy array).

        Returns:
            (peace_detected, all_hand_landmarks)
            all_hand_landmarks is a list of landmark lists (empty if no hands).
        """
        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += int(1000 / 30)  # ~33ms per frame at 30fps

        peace_detected = False
        all_hands = list(result.hand_landmarks) if result.hand_landmarks else []

        for hand_lm in all_hands:
            if self.is_peace_sign(hand_lm):
                peace_detected = True
                break

        return peace_detected, all_hands

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
