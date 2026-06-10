import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import winsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = r"D:\FINAL PRJ\drowsiness_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)

EAR_LOW_THRESHOLD = 0.20
EAR_VERY_LOW_THRESHOLD = 0.16
CNN_DROWSY_THRESHOLD = 0.90
CNN_STRONG_DROWSY_THRESHOLD = 0.95

SCORE_ALERT_THRESHOLD = 12
MAX_SCORE = 15
BEEP_COOLDOWN = 1.0
ALERT_CONFIRM_SECONDS = 2.0
STRONG_DROWSY_CONFIRM_SECONDS = 0.8
OPEN_EYE_RESET_EAR = 0.22
SIDE_HEAD_YAW_THRESHOLD = 25
HEAD_DOWN_PITCH_THRESHOLD = 18


def get_beep_cooldown(score):
    if score >= 15:
        return 0.20
    if score >= 14:
        return 0.30
    if score >= 12:
        return 0.50
    return BEEP_COOLDOWN


model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CROP = [33, 133, 160, 158, 153, 144, 159, 145]
RIGHT_EYE_CROP = [362, 263, 385, 387, 373, 380, 386, 374]

HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def crop_eye(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(min(xs) - 20, 0)
    x_max = min(max(xs) + 20, w)
    y_min = max(min(ys) - 20, 0)
    y_max = min(max(ys) + 20, h)

    eye_crop = frame[y_min:y_max, x_min:x_max]

    if eye_crop.size == 0:
        return None, (x_min, y_min, x_max, y_max)

    return eye_crop, (x_min, y_min, x_max, y_max)


def predict_eye_drowsy(eye_crop):
    if eye_crop is None:
        return 0.0

    eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
    eye_resized = cv2.resize(eye_rgb, IMAGE_SIZE)

    eye_array = np.expand_dims(eye_resized.astype(np.float32), axis=0)
    eye_array = preprocess_input(eye_array)

    pred = model.predict(eye_array, verbose=0)[0][0]

    # With class_indices {'drowsy': 0, 'non_drowsy': 1},
    # sigmoid near 0 means drowsy, near 1 means non_drowsy.
    return 1.0 - float(pred)


def estimate_head_pose(landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array(
        [
            (landmarks[HEAD_POSE_LANDMARKS["nose_tip"]].x * w, landmarks[HEAD_POSE_LANDMARKS["nose_tip"]].y * h),
            (landmarks[HEAD_POSE_LANDMARKS["chin"]].x * w, landmarks[HEAD_POSE_LANDMARKS["chin"]].y * h),
            (landmarks[HEAD_POSE_LANDMARKS["left_eye_corner"]].x * w, landmarks[HEAD_POSE_LANDMARKS["left_eye_corner"]].y * h),
            (landmarks[HEAD_POSE_LANDMARKS["right_eye_corner"]].x * w, landmarks[HEAD_POSE_LANDMARKS["right_eye_corner"]].y * h),
            (landmarks[HEAD_POSE_LANDMARKS["left_mouth_corner"]].x * w, landmarks[HEAD_POSE_LANDMARKS["left_mouth_corner"]].y * h),
            (landmarks[HEAD_POSE_LANDMARKS["right_mouth_corner"]].x * w, landmarks[HEAD_POSE_LANDMARKS["right_mouth_corner"]].y * h),
        ],
        dtype=np.float64,
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

drowsiness_score = 0
last_beep_time = 0
alert_start_time = None
while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "No face"
    color = (255, 255, 255)
    avg_ear = 0.0
    cnn_drowsy_prob = 0.0
    pitch = 0.0
    yaw = 0.0
    roll = 0.0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        pitch, yaw, roll = estimate_head_pose(landmarks, frame.shape)
        looking_side = abs(yaw) > SIDE_HEAD_YAW_THRESHOLD
        looking_down = pitch > HEAD_DOWN_PITCH_THRESHOLD

        left_eye_points = []
        right_eye_points = []

        for idx in LEFT_EYE:
            lm = landmarks[idx]
            left_eye_points.append((int(lm.x * w), int(lm.y * h)))

        for idx in RIGHT_EYE:
            lm = landmarks[idx]
            right_eye_points.append((int(lm.x * w), int(lm.y * h)))

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        left_crop, left_box = crop_eye(frame, landmarks, LEFT_EYE_CROP)
        right_crop, right_box = crop_eye(frame, landmarks, RIGHT_EYE_CROP)

        left_prob = predict_eye_drowsy(left_crop)
        right_prob = predict_eye_drowsy(right_crop)
        cnn_drowsy_prob = max(left_prob, right_prob)

        for box in [left_box, right_box]:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        if looking_side:
            if avg_ear < EAR_VERY_LOW_THRESHOLD and cnn_drowsy_prob > CNN_STRONG_DROWSY_THRESHOLD:
                drowsiness_score += 2
            else:
                drowsiness_score -= 1
        else:
            if avg_ear < EAR_VERY_LOW_THRESHOLD:
                drowsiness_score += 2
            elif avg_ear < EAR_LOW_THRESHOLD:
                drowsiness_score += 1
            else:
                drowsiness_score -= 1

        if looking_side:
            if cnn_drowsy_prob > 0.97:
                drowsiness_score += 2
        else:
            if cnn_drowsy_prob > CNN_DROWSY_THRESHOLD:
                drowsiness_score += 5
            elif cnn_drowsy_prob > 0.60:
                drowsiness_score += 2

        if looking_down:
            drowsiness_score += 2

        drowsiness_score = max(0, min(drowsiness_score, MAX_SCORE))

        current_frame_drowsy = (
            (avg_ear < EAR_LOW_THRESHOLD and not looking_side)
            or cnn_drowsy_prob > CNN_STRONG_DROWSY_THRESHOLD
            or looking_down
        )

        current_frame_open = (
            avg_ear > OPEN_EYE_RESET_EAR
            and cnn_drowsy_prob < 0.60
        )

        if current_frame_open:
            alert_start_time = None

        if drowsiness_score >= SCORE_ALERT_THRESHOLD:
            current_time = time.time()

            if alert_start_time is None:
                if current_frame_drowsy:
                    alert_start_time = current_time

            alert_duration = 0 if alert_start_time is None else current_time - alert_start_time
            required_confirm_seconds = ALERT_CONFIRM_SECONDS

            if cnn_drowsy_prob > CNN_STRONG_DROWSY_THRESHOLD:
                required_confirm_seconds = STRONG_DROWSY_CONFIRM_SECONDS

            if alert_duration >= required_confirm_seconds:
                status_text = "ALERT: DROWSY"
                color = (0, 0, 255)

                beep_cooldown = get_beep_cooldown(drowsiness_score)
                if cnn_drowsy_prob > CNN_STRONG_DROWSY_THRESHOLD:
                    beep_cooldown = min(beep_cooldown, 0.30)

                if current_time - last_beep_time >= beep_cooldown:
                    winsound.Beep(1200, 150)
                    last_beep_time = current_time
            else:
                status_text = f"Warning {alert_duration:.1f}s"
                color = (0, 255, 255)

        elif drowsiness_score >= 7:
            alert_start_time = None
            status_text = "Warning"
            color = (0, 255, 255)
        else:
            alert_start_time = None
            status_text = "Normal"
            color = (0, 255, 0)

        for p in left_eye_points + right_eye_points:
            cv2.circle(frame, p, 2, (0, 255, 0), -1)

    else:
        drowsiness_score = max(0, drowsiness_score - 1)
        alert_start_time = None

    cv2.putText(
        frame,
        f"Status: {status_text}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"Score: {drowsiness_score}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"EAR: {avg_ear:.3f}",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"CNN drowsy prob: {cnn_drowsy_prob:.2f}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}",
        (30, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Real-time Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
