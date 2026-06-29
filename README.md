# foto-kita-blur

Gesture-controlled camera blur using Python + MediaPipe.

Make a ✌️ peace sign at your webcam — the entire frame smoothly blurs out. Remove the gesture and it fades back to clear.

## How it works

- **MediaPipe Hands** detects 21 hand landmarks in real-time
- Peace sign = index + middle fingers extended, ring + pinky folded
- **Gaussian blur** applied to the full frame with smooth lerp transition
- **Hand skeleton overlay** with color-coded fingertips (green = extended, orange = folded)

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

## Tech

Python · Flask · OpenCV · MediaPipe
