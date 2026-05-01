import streamlit as st
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from collections import defaultdict
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- 1. SETTINGS & SETUP ---
SAVED_FRAMES_DIR = "saved_frames"
if not os.path.exists(SAVED_FRAMES_DIR):
    os.makedirs(SAVED_FRAMES_DIR)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 2. FULL CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Dancing+Script:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #ffe6f0 0%, #e6d5ff 50%, #d5e8ff 100%);
        padding-top: 60px !important;
    }
    
    .top-ribbon {
        position: fixed; top: 0; left: 0; right: 0; text-align: center; font-size: 24px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 10px; color: white; font-family: 'Dancing Script', cursive;
        z-index: 999; box-shadow: 0 2px 10px rgba(0,0,0,0.1); letter-spacing: 5px;
    }
    
    .bottom-ribbon {
        position: fixed; bottom: 0; left: 0; right: 0; text-align: center; font-size: 20px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 8px; color: white; font-family: 'Dancing Script', cursive;
        z-index: 999; letter-spacing: 5px;
    }

    html, body, [class*="st-"] {
        font-family: 'Quicksand', sans-serif !important;
    }

    h1 {
        font-family: 'Dancing Script', cursive !important;
        font-size: 3.5em !important;
        text-align: center !important;
        background: linear-gradient(135deg, #ff69b4, #9370db, #4169e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stButton > button {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(12px) !important;
        border: 2px solid rgba(255, 182, 193, 0.6) !important;
        border-radius: 25px !important;
        color: #6b3e6b !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 240, 245, 0.85) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 182, 193, 0.5) !important;
    }
</style>

<div class="top-ribbon">✨🌸✨ Object Detection ✨🌸✨</div>
<div class="bottom-ribbon">✨ Made with 💕 by AI Magic ✨</div>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Live Object Detection & Tracing ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b3e6b; font-size: 1.2em;'>🌸 Real-time AI magic in your browser 🌸</p>", unsafe_allow_html=True)

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ✨💖 Settings 💖✨")
    mirror_view = st.checkbox("🪞 Mirror View", value=True)
    show_counting = st.checkbox("🔢 Show Object Counting", value=True)
    conf_thresh = st.slider("🎯 Confidence", 0.0, 1.0, 0.4)
    
    if st.button("🔄 Reset All", use_container_width=True):
        st.rerun()

# --- 4. VIDEO PROCESSOR CLASS ---
class YOLOProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.counts = defaultdict(int)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if mirror_view:
            img = cv2.flip(img, 1)

        # Inference
        results = self.model.track(img, persist=True, conf=conf_thresh, verbose=False)
        
        # Draw Results
        annotated_frame = results[0].plot()
        
        # Update Counts for the UI
        self.counts = results[0].boxes.cls.unique().tolist() if results[0].boxes is not None else []
        self.class_names = results[0].names
        
        # Add simple FPS/Resolution Overlay
        cv2.putText(annotated_frame, f"AI Active", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)

        return annotated_frame

# --- 5. MAIN UI LAYOUT ---
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    ctx = webrtc_streamer(
        key="yolo-girly-app",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("---")
disp_col1, disp_col2 = st.columns(2)

with disp_col1:
    st.markdown("#### 📊 Object Analysis")
    if ctx.video_processor:
        # This pulls data from the video thread back to the Streamlit UI
        found_classes = ctx.video_processor.counts
        names = ctx.video_processor.class_names
        if found_classes:
            for c in found_classes:
                st.write(f"✨ **Detected:** {names[int(c)]}")
        else:
            st.write("Searching for objects... 🔍")

with disp_col2:
    st.markdown("#### 📸 Quick Actions")
    if st.button("🖼️ Save Current View"):
        st.info("To save a frame in WebRTC mode, right-click the video and select 'Save Image As'.")

# --- 6. GALLERY ---
st.markdown("### 💾 Saved Frames")
saved_frames = [f for f in os.listdir(SAVED_FRAMES_DIR) if f.endswith(".jpg")]
if saved_frames:
    cols = st.columns(3)
    for idx, frame in enumerate(saved_frames[-3:]):
        cols[idx].image(os.path.join(SAVED_FRAMES_DIR, frame), use_container_width=True)
else:
    st.write("💝 No frames saved yet.")
