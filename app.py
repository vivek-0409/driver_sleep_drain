import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS (Professional UI + Animation)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.card {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(59,130,246,0.3);
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}

.title {
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #cbd5f5;
}

.step {
    font-size: 1rem;
    padding: 10px 0;
}

.status-box {
    font-size: 1.4rem;
    font-weight: 700;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}

.alert {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
}

.drowsy {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("<div class='title'>🚗 Driver Drowsiness Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered eye-based drowsiness monitoring using MediaPipe</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# Step-by-Step Guide
# -------------------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ How to Use (Step-by-Step)")
    st.markdown("""
    <div class='step'>1️⃣ Upload a **driving video** 🎥</div>
    <div class='step'>2️⃣ Click **▶ Start Detection**</div>
    <div class='step'>3️⃣ Watch real-time detection status</div>
    <div class='step'>4️⃣ Click **⏹ Stop Detection** anytime</div>
    """, unsafe_allow_html=True)
    st.markdown("</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# MediaPipe Setup
# -------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EYE_RATIO_THRESHOLD = 0.20

def eye_ratio(landmarks, eye, w, h):
    pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2 * h_dist)

# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
video_file = st.file_uploader("📤 Upload Video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
st.markdown("</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# Controls
# -------------------------------------------------
col1, col2 = st.columns(2)
start = col1.button("▶ Start Detection", use_container_width=True)
stop = col2.button("⏹ Stop Detection", use_container_width=True)

frame_box = st.image([])
status_placeholder = st.empty()

if video_file and start:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video_file.read())
    cap = cv2.VideoCapture(temp.name)

    st.session_state["run"] = True

    while cap.isOpened() and st.session_state.get("run", True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        status = "NO FACE"
        box_class = "alert"

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            left = eye_ratio(lm, LEFT_EYE, w, h)
            right = eye_ratio(lm, RIGHT_EYE, w, h)
            avg = (left + right) / 2

            if avg < EYE_RATIO_THRESHOLD:
                status = "DROWSY 😴"
                box_class = "drowsy"
            else:
                status = "ALERT 😀"
                box_class = "alert"

        frame_box.image(frame, channels="BGR")
        status_placeholder.markdown(
            f"<div class='status-box {box_class}'>Status: {status}</div>",
            unsafe_allow_html=True
        )

        time.sleep(0.03)

    cap.release()

if stop:
    st.session_state["run"] = False
    st.warning("⏹ Detection stopped by user.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<br><center>⚠️ Live webcam is not supported on Streamlit Cloud. Use local/VPS for real camera.</center>", unsafe_allow_html=True)
