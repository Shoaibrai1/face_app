import streamlit as st
import cv2
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# Page config
st.set_page_config(page_title="Face Auth App", layout="centered")

# Inject custom CSS
st.markdown("""
<style>
    body {
        background-color: #f9f9f9;
    }
    .main {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        margin: 10px auto;
        display: block;
    }
    .stRadio > div {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎥 Face Signup & Login App")

DATASET_PATH = "dataset"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# Face matching function
def match_face(captured_face):
    saved_path = os.path.join(DATASET_PATH, 'user.jpg')
    if not os.path.exists(saved_path):
        return False, "No registered user"
    saved_face = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
    if saved_face is None:
        return False, "Saved face is corrupted"
    captured_face = cv2.resize(captured_face, (saved_face.shape[1], saved_face.shape[0]))
    diff = cv2.absdiff(saved_face, captured_face)
    score = np.sum(diff)
    return (score < 100000), f"Score: {score}"

# Detect face from frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return gray[y:y+h, x:x+w]
    return None

# WebRTC class
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

st.markdown("<h3 style='text-align:center;'>Please choose a mode:</h3>", unsafe_allow_html=True)
mode = st.radio("Choose Mode", ["Signup", "Login"], horizontal=True, label_visibility="collapsed")


# Start video
webrtc_ctx = webrtc_streamer(
    key="face-cam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    frame = webrtc_ctx.video_processor.frame
    if frame is not None:
        st.image(frame, channels="BGR", caption="Live Camera Feed")

        if st.button("📸 Capture Face"):
            with st.spinner("Processing your face..."):
                face = process_frame(frame)
                if face is None:
                    st.error("😞 No face detected. Try again.")
                else:
                    if mode == "Signup":
                        cv2.imwrite(os.path.join(DATASET_PATH, 'user.jpg'), face)
                        st.success("✅ Face saved! You can now login.")
                    elif mode == "Login":
                        matched, msg = match_face(face)
                        if matched:
                            st.session_state['is_logged_in'] = True
                            st.success("🎉 Login successful!")
                        else:
                            st.error("🚫 Face not recognized. Try again.")
                            st.caption(msg)

# After login UI
if st.session_state['is_logged_in']:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🏠 Welcome, Registered User")
    st.write("You have successfully logged in using face recognition.")
    if st.button("🔒 Logout"):
        st.session_state['is_logged_in'] = False
        st.experimental_rerun()
