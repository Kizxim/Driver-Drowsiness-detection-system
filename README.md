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

## How It Works

The system combines MobileNetV2 eye-state classification with rule-based computer vision logic.

OpenCV captures webcam frames, while MediaPipe FaceMesh detects facial landmarks and eye regions. The system calculates Eye Aspect Ratio (EAR) to estimate eye openness, then classifies cropped eye images as drowsy or non-drowsy using a MobileNetV2-based model.

A temporal scoring mechanism is used instead of single-frame prediction, reducing false alerts from normal blinking. Head-pose cues are also considered to handle side movement and downward gaze. When the drowsiness score stays high for a sustained period, an audio warning alert is triggered.
```
## Model Training

The eye-state classifier was trained on a balanced drowsy/non-drowsy eye image dataset using transfer learning with MobileNetV2.

The training pipeline includes:

- Train/validation/test split
- Image preprocessing and normalization
- Data augmentation
- Frozen MobileNetV2 feature extractor
- Dense classification layers
- Dropout for regularization
- Early stopping and model checkpointing

## Real-Time Testing

The system was manually tested across common webcam scenarios:

- Normal open eyes
- Normal blinking
- Sustained eye closure
- Side head movement
- Downward gaze
- Different distances from the webcam

The system performs most reliably when the user faces the camera at a normal laptop distance. False warnings may occur when the face is too far from the webcam, when the eyes are partially occluded, or when head angle affects eye landmark detection.

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```
```md
## Current Limitations

- Performance may decrease when the face is too far from the webcam.
- Downward gaze can sometimes be detected as drowsiness because the eyelid distance becomes smaller.
- Lighting condition and partial eye occlusion can affect landmark detection.
├── requirements.txt                # Python dependencies
└── README.md
