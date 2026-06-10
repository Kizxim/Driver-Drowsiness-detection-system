# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using webcam input, OpenCV, MediaPipe FaceMesh, and a MobileNetV2-based eye-state classifier.

The system detects eye regions from webcam frames, predicts whether the eyes are drowsy or non-drowsy, combines the result with Eye Aspect Ratio (EAR) and head-pose cues, then triggers warning alerts when drowsiness is detected for a sustained period.

## Features

- Real-time webcam processing with OpenCV
- Face and eye landmark detection using MediaPipe FaceMesh
- Eye Aspect Ratio (EAR) calculation for eye openness estimation
- MobileNetV2-based transfer learning model for drowsy/non-drowsy eye classification
- Temporal scoring logic to reduce false alerts from normal blinking
- Head-pose-aware rule adjustment for side movement and downward gaze
- Audio warning alert when sustained drowsiness is detected

## Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
.
├── realtime_drowsiness.py          # Real-time webcam drowsiness detection
├── train_drowsiness.py             # Model training and evaluation script
├── drowsiness_mobilenetv2.keras    # Trained eye-state classification model
├── eye_model_training_curve.png    # Training accuracy/loss curve
├── requirements.txt                # Python dependencies
└── README.md
