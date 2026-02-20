import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open the camera
cap = cv2.VideoCapture(0)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C)

# ---- Sleep detection variables ----
eye_closed_threshold = 0.25
sleep_duration = 2   # seconds
eye_closed_start = None
sleeping_detected = False

# ---- Yawn detection variables ----
yawn_threshold = 10
yawn_duration = 2
yawn_start = None
yawn_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]

            # Eye landmarks
            left_eye = [points[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [points[i] for i in [362, 385, 386, 263, 373, 380]]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ---- Sleep detection ----
            if ear < eye_closed_threshold:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= sleep_duration:
                    sleeping_detected = True
            else:
                eye_closed_start = None
                sleeping_detected = False

            # ---- Yawn detection ----
            upper_lip = points[13][1]
            lower_lip = points[14][1]
            mouth_open = abs(lower_lip - upper_lip)

            if mouth_open > yawn_threshold:
                if yawn_start is None:
                    yawn_start = time.time()
                elif time.time() - yawn_start >= yawn_duration:
                    yawn_detected = True
            else:
                yawn_start = None
                yawn_detected = False

            # ---- Display alerts ----
            if sleeping_detected:
                cv2.putText(frame, "SLEEPING DETECTED",
                            (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

            if yawn_detected:
                cv2.putText(frame, "YAWNING DETECTED",
                            (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 3)

    cv2.imshow("Sleep & Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()