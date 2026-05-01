import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- 1. SESSION STATE & DIRECTORIES ---
SAVED_FRAMES_DIR = "saved_frames"
if not os.path.exists(SAVED_FRAMES_DIR):
    os.makedirs(SAVED_FRAMES_DIR)

if 'object_counts' not in st.session_state:
    st.session_state.object_counts = defaultdict(int)
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 3. THE "GIRLY RIBBON" CSS (UNCHANGED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Dancing+Script:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #ffe6f0 0%, #e6d5ff 50%, #d5e8ff 100%);
        padding-top: 60px !important;
        padding-bottom: 60px !important;
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

    html, body, .stMarkdown, .stText, .stButton, .stCheckbox, .stSelectbox, .stSlider {
        font-family: 'Quicksand', sans-serif !important;
    }

    h1 {
        font-family: 'Dancing Script', cursive !important;
        font-size: 3.5em !important;
        text-align: center !important;
        background: linear-gradient(135deg, #ff69b4, #9370db, #4169e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(12px) !important;
        border: 2px solid rgba(255, 182, 193, 0.6) !important;
        color: #6b3e6b !important;
        border-radius: 25px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 240, 245, 0.85) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 182, 193, 0.5) !important;
    }
</style>

<div class="top-ribbon">✨🌸✨ Object Detection ✨🌸✨</div>
<div class="bottom-ribbon">✨ Made with 💕 by AI Magic | Object Detection & Tracking ✨</div>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Live Object Detection & Tracing ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b3e6b; font-size: 1.2em;'>🌸 Point your camera at objects to identify them in real-time with AI magic 🌸</p>", unsafe_allow_html=True)

# --- 4. SIDEBAR SETTINGS ---
with st.sidebar:
    st.markdown("### ✨💖 Settings 💖✨")
    mirror_view = st.checkbox("🪞 Mirror View (Inverted)", value=True)
    show_counting = st.checkbox("🔢 Show Object Counting", value=True)
    enable_alerts = st.checkbox("🔔 Enable Alerts", value=True)
    conf_thresh = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.4)
    
    if st.button("🔄 Reset All Counters", use_container_width=True):
        st.session_state.object_counts.clear()
        st.session_state.detection_log.clear()
        st.rerun()

# --- 5. WEBRTC VIDEO PROCESSOR ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if mirror_view:
            img = cv2.flip(img, 1)

        results = self.model.track(img, persist=True, conf=conf_thresh, verbose=False)
        
        # This draws the pretty boxes and labels automatically
        annotated_frame = results[0].plot()
        
        # Update session data (Classes detected)
        if results[0].boxes is not None:
            self.current_classes = [results[0].names[int(c)] for c in results[0].boxes.cls]
        else:
            self.current_classes = []

        return annotated_frame

# --- 6. MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    ctx = webrtc_streamer(
        key="yolo-live",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("---")
disp_col1, disp_col2, disp_col3 = st.columns(3)

with disp_col1:
    st.markdown("#### 📊 Object Analysis")
    count_placeholder = st.empty()
    if ctx.video_processor:
        detected = ctx.video_processor.current_classes
        if detected:
            counts = defaultdict(int)
            for x in detected: counts[x] += 1
            count_text = "".join([f"**{k}:** {v} \n\n" for k, v in counts.items()])
            count_placeholder.markdown(count_text)
        else:
            count_placeholder.write("No objects detected yet 🔍")

with disp_col2:
    st.markdown("#### 🚨 Alerts")
    if ctx.video_processor and enable_alerts:
        detected = list(set(ctx.video_processor.current_classes))
        if detected:
            st.warning(f"🔔 Detected: {', '.join(detected)}")

with disp_col3:
    st.markdown("#### 💾 Saved Frames")
    saved_frames = [f for f in os.listdir(SAVED_FRAMES_DIR) if f.endswith(".jpg")]
    if saved_frames:
        st.write(f"📸 Total saved: {len(saved_frames)}")
        for frame_file in saved_frames[-3:]:
            with st.expander(f"🖼️ {frame_file[:15]}..."):
                st.image(os.path.join(SAVED_FRAMES_DIR, frame_file))
    else:
        st.write("💝 No frames saved yet")
