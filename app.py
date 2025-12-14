# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import collections
import threading
import time
from playsound import playsound

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Driver Drowsiness Monitor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Sleep / Drowsiness Detection System")
st.markdown("Real-time eye monitoring using **MediaPipe + OpenCV**")

# -----------------------------
# Parameters
# -----------------------------
CAMERA_ID = 0
BUFFER_LEN = 150
ACTIVE_PERCENT_THRESHOLD = 70
EYE_RATIO_THRESHOLD = 0.20
SIREN_FILE = "Alarm.wav"

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks
L_EYE_UP, L_EYE_DOWN, L_EYE_LEFT, L_EYE_RIGHT = 159, 145, 33, 133
R_EYE_UP, R_EYE_DOWN, R_EYE_LEFT, R_EYE_RIGHT = 386, 374, 362, 263

recent_activity = collections.deque(maxlen=BUFFER_LEN)

alarm_playing = threading.Event()
alarm_stop = threading.Event()

# -----------------------------
# Helper functions
# -----------------------------
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def landmark_to_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def compute_eye_ratio(landmarks, w, h, up, down, left, right):
    up = landmark_to_point(landmarks[up], w, h)
    down = landmark_to_point(landmarks[down], w, h)
    left = landmark_to_point(landmarks[left], w, h)
    right = landmark_to_point(landmarks[right], w, h)

    vert = euclidean(up, down)
    hor = euclidean(left, right)
    return 0 if hor == 0 else vert / hor

# -----------------------------
# Alarm Thread
# -----------------------------
def alarm_worker():
    while not alarm_stop.is_set():
        alarm_playing.wait()
        if alarm_stop.is_set():
            break
        try:
            playsound(SIREN_FILE)
        except:
            pass

threading.Thread(target=alarm_worker, daemon=True).start()

# -----------------------------
# Streamlit Controls
# -----------------------------
run = st.checkbox("▶️ Start Camera")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# -----------------------------
# Main Loop
# -----------------------------
if run:
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        st.error("❌ Camera not accessible")
    else:
        st.success("✅ Camera Started")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Camera frame not received")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            frame_active = False
            eye_ratio = 1.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                l_ratio = compute_eye_ratio(landmarks, w, h, L_EYE_UP, L_EYE_DOWN, L_EYE_LEFT, L_EYE_RIGHT)
                r_ratio = compute_eye_ratio(landmarks, w, h, R_EYE_UP, R_EYE_DOWN, R_EYE_LEFT, R_EYE_RIGHT)
                eye_ratio = (l_ratio + r_ratio) / 2
                frame_active = eye_ratio > EYE_RATIO_THRESHOLD

            recent_activity.append(1 if frame_active else 0)
            active_percent = (sum(recent_activity) / len(recent_activity)) * 100

            if len(recent_activity) == BUFFER_LEN:
                if active_percent < ACTIVE_PERCENT_THRESHOLD:
                    alarm_playing.set()
                else:
                    alarm_playing.clear()

            status = "🟢 ACTIVE" if frame_active else "🔴 INACTIVE"

            status_placeholder.markdown(
                f"""
                **Eye Ratio:** `{eye_ratio:.3f}`  
                **Status:** {status}  
                **Active %:** `{active_percent:.1f}%`
                """
            )

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.03)

        cap.release()
        alarm_stop.set()
        alarm_playing.clear()
