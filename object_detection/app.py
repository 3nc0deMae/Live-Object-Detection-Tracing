import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
from collections import defaultdict
from datetime import datetime
import os
import time
import numpy as np

# Create saved_frames folder if it doesn't exist
SAVED_FRAMES_DIR = "saved_frames"
if not os.path.exists(SAVED_FRAMES_DIR):
    os.makedirs(SAVED_FRAMES_DIR)

# Initialize session state
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = defaultdict(int)
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = 0
if 'last_annotated_frame' not in st.session_state:
    st.session_state.last_annotated_frame = None
if 'resolution' not in st.session_state:
    st.session_state.resolution = "640x480"
if 'webrtc_ctx' not in st.session_state:
    st.session_state.webrtc_ctx = None
if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0
if 'last_auto_save_time' not in st.session_state:
    st.session_state.last_auto_save_time = 0
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'fps_update_time' not in st.session_state:
    st.session_state.fps_update_time = time.time()
if 'current_fps' not in st.session_state:
    st.session_state.current_fps = 0

# Cache the model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    # Warm up the model
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_input, verbose=False)
    return model

model = load_model()

# Custom CSS for girly ribbon theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Dancing+Script:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #ffe6f0 0%, #e6d5ff 50%, #d5e8ff 100%);
        padding-top: 60px !important;
        padding-bottom: 60px !important;
    }
    
    .top-ribbon {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 24px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 10px;
        color: white;
        font-family: 'Dancing Script', cursive;
        letter-spacing: 5px;
        z-index: 999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .bottom-ribbon {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 20px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 8px;
        color: white;
        font-family: 'Dancing Script', cursive;
        letter-spacing: 5px;
        z-index: 999;
    }
    
    html, body, .stMarkdown, .stText, .stButton, .stCheckbox, .stSelectbox, .stSlider {
        font-family: 'Quicksand', sans-serif !important;
    }
    
    h1, h2, h3, .stSubheader {
        font-family: 'Dancing Script', cursive !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #ff69b4, #9370db, #4169e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h1 {
        font-size: 3.5em !important;
        margin-top: 0px !important;
        text-align: center !important;
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(12px) !important;
        border: 2px solid rgba(255, 182, 193, 0.6) !important;
        color: #6b3e6b !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border-radius: 25px !important;
        padding: 10px 25px !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.55) !important;
        border: 2px solid #ffb6c1 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(255, 105, 180, 0.3) !important;
        color: #8b3e8b !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(255, 182, 193, 0.6), rgba(221, 160, 221, 0.6), rgba(135, 206, 235, 0.6)) !important;
        border: 2px solid #ffb6c1 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 240, 245, 0.85) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 20px !important;
        margin: 10px !important;
        border: 2px solid rgba(255, 182, 193, 0.5) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #6b3e6b !important;
        font-family: 'Quicksand', sans-serif !important;
    }
    
    .streamlit-expanderHeader span:first-child {
        display: none !important;
    }
    
    .streamlit-expanderHeader svg {
        display: inline-block !important;
    }
    
    .streamlit-expanderHeader::before,
    .streamlit-expanderHeader::after {
        display: none !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        color: #6b3e6b !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 182, 193, 0.5) !important;
    }
    
    .stCheckbox > label {
        color: #6b3e6b !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > label {
        color: #6b3e6b !important;
        font-weight: 500 !important;
    }
    
    .stSlider > label {
        color: #6b3e6b !important;
        font-weight: 500 !important;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        color: #6b3e6b !important;
        border-left: 4px solid #ffb6c1 !important;
    }
    
    .stColumn > div {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 20px !important;
        padding: 15px !important;
        border: 1px solid rgba(255, 182, 193, 0.4) !important;
    }
    
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(221, 160, 221, 0.6) !important;
        color: #6b3e6b !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #ffe6f0;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ffb6c1, #dda0dd, #87ceeb);
        border-radius: 10px;
    }
    
    .stButton > button, .streamlit-expanderHeader {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .streamlit-webrtc video {
        border-radius: 20px !important;
        width: 100% !important;
        height: auto !important;
        background: rgba(0,0,0,0.1) !important;
    }
    
    .stVideo {
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 10px !important;
    }
    
    /* Hide the default WebRTC start/stop buttons */
    .streamlit-webrtc .stButton {
        display: none !important;
    }
    
    /* Hide any extra button elements */
    .streamlit-webrtc button {
        display: none !important;
    }
</style>

<div class="top-ribbon">
    ✨🌸✨Object Detection ✨🌸✨
</div>

<div class="bottom-ribbon">
    ✨ Made with 💕 by AI Magic | Object Detection & Tracking ✨
</div>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Live Object Detection & Tracing ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b3e6b; font-size: 1.2em; font-family: Quicksand; margin-bottom: 30px;'>🌸 Point your camera at objects to identify them in real-time with AI magic 🌸</p>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### ✨💖 Settings 💖✨")
    
    st.markdown("#### 🎥 Camera Settings")
    mirror_view = st.checkbox("🪞 Mirror View (Inverted)", value=True)
    
    st.markdown("#### 📱 Quality & Resolution")
    resolution_options = {
        "Low (480p) - Fastest": "640x480",
        "Medium (720p) - Balanced": "1280x720", 
        "High (1080p) - High Quality": "1920x1080",
    }
    selected_resolution = st.selectbox(
        "🎬 Select Resolution",
        options=list(resolution_options.keys()),
        index=0,
        help="Higher resolution = better quality but more processing power needed"
    )
    st.session_state.resolution = resolution_options[selected_resolution]
    
    if "1080" in st.session_state.resolution:
        st.info("💕 1080p High Quality - Great detail, good performance")
    elif "720" in st.session_state.resolution:
        st.info("✨ 720p Balanced - Good quality and speed")
    else:
        st.success("🌸 480p Fast mode - Optimized for smooth performance")
    
    st.markdown("#### 📊 Object Counting")
    show_counting = st.checkbox("🔢 Show Object Counting", value=True)
    
    st.markdown("#### 🚨 Alert System")
    enable_alerts = st.checkbox("🔔 Enable Alerts", value=True)
    alert_objects = st.multiselect(
        "🎯 Alert for these objects",
        options=["person", "cell phone", "bottle", "laptop", "chair", "book", "tv", "cat", "dog", "bird"],
        default=["person"]
    )
    
    st.markdown("#### 💾 Frame Saving")
    save_frame_request = st.button("📸 Save Current Frame", use_container_width=True)
    auto_save = st.checkbox("🤖 Auto-save every 10 seconds", value=False)
    
    if st.button("🔄 Reset All Counters", use_container_width=True):
        st.session_state.object_counts.clear()
        st.session_state.detection_log.clear()
        st.success("✨ Counters reset successfully! ✨")
    
    st.markdown("---")
    st.markdown("#### 🗑️ Manage Saved Frames")
    if st.button("🗑️ Delete ALL Saved Frames", use_container_width=True):
        saved_frames = [f for f in os.listdir(SAVED_FRAMES_DIR) if f.startswith(("detected_frame_", "auto_saved_frame_")) and f.endswith(".jpg")]
        if saved_frames:
            for frame in saved_frames:
                try:
                    os.remove(os.path.join(SAVED_FRAMES_DIR, frame))
                except:
                    pass
            st.success(f"✅ Deleted {len(saved_frames)} saved frames 💕")
            st.rerun()
        else:
            st.info("💝 No saved frames to delete")

# Video display area
video_display_area = st.empty()

# Camera control buttons (only one set of buttons)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not st.session_state.camera_active:
        if st.button("📷 Start Camera 💖", use_container_width=True, type="primary"):
            st.session_state.camera_active = True
            st.session_state.object_counts.clear()
            st.session_state.detection_log.clear()
            st.rerun()
    else:
        if st.button("⏹️ Stop Camera 💔", use_container_width=True, type="secondary"):
            st.session_state.camera_active = False
            st.session_state.webrtc_ctx = None
            video_display_area.empty()
            st.rerun()

# Display area for counts and alerts
st.markdown("---")
disp_col1, disp_col2, disp_col3 = st.columns(3)

with disp_col1:
    st.markdown("#### 📊 Object Count")
    count_placeholder = st.empty()
    
with disp_col2:
    st.markdown("#### 🚨 Recent Alerts")
    alert_placeholder = st.empty()
    
with disp_col3:
    st.markdown("#### 💾 Saved Frames")
    saved_frames = [f for f in os.listdir(SAVED_FRAMES_DIR) if f.startswith(("detected_frame_", "auto_saved_frame_")) and f.endswith(".jpg")]
    saved_frames.sort(reverse=True)
    
    if saved_frames:
        st.write(f"📸 Total saved: {len(saved_frames)} frames")
        for idx, frame_file in enumerate(saved_frames[:5]):
            frame_path = os.path.join(SAVED_FRAMES_DIR, frame_file)
            with open(frame_path, "rb") as file:
                with st.expander(f"🖼️ {frame_file[:30]}..."):
                    img = cv2.imread(frame_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(
                            label="📥 Download",
                            data=file,
                            file_name=frame_file,
                            mime="image/jpeg",
                            key=f"download_{frame_file}_{idx}"
                        )
                    with col_btn2:
                        if st.button(f"🗑️ Delete", key=f"delete_{frame_file}_{idx}"):
                            try:
                                os.remove(frame_path)
                                st.success(f"✅ Deleted: {frame_file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
    else:
        st.write("💝 No frames saved yet")

# Object colors
OBJECT_COLORS = {
    'person': (255, 105, 180),
    'cell phone': (135, 206, 235),
    'laptop': (100, 149, 237),
    'tv': (70, 130, 180),
    'mouse': (173, 216, 230),
    'keyboard': (0, 191, 255),
    'chair': (221, 160, 221),
    'couch': (218, 112, 214),
    'bed': (186, 85, 211),
    'dining table': (153, 50, 204),
    'bottle': (144, 238, 144),
    'cup': (152, 251, 152),
    'book': (255, 182, 193),
    'clock': (255, 228, 196),
    'cat': (255, 215, 0),
    'dog': (255, 140, 0),
    'bird': (255, 165, 0),
    'horse': (255, 69, 0),
    'car': (255, 99, 71),
    'bicycle': (255, 127, 80),
    'motorcycle': (255, 160, 122),
    'bus': (255, 20, 147),
    'truck': (255, 105, 180),
    'default': (147, 112, 219)
}

def get_object_color(class_name):
    return OBJECT_COLORS.get(class_name.lower(), OBJECT_COLORS['default'])

def draw_smooth_boxes(frame, boxes_data):
    frame_copy = frame.copy()
    for box_data in boxes_data:
        box = box_data['box']
        class_name = box_data['class']
        confidence = box_data['confidence']
        
        x1, y1, x2, y2 = map(int, box)
        color = get_object_color(class_name)
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {confidence:.2f}" if confidence else class_name
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(frame_copy, (x1, y1 - label_h - 5), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 2, y1 - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame_copy

def add_overlays(frame, object_counts, mirror_view):
    frame_copy = frame.copy()
    
    if show_counting and object_counts:
        active_counts = {k: v for k, v in object_counts.items() if v > 0}
        if active_counts:
            y_offset = 25
            for obj, count in list(active_counts.items())[:5]:
                cv2.putText(frame_copy, f"{obj}: {count}", 
                           (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.45, (255, 105, 180), 1)
                y_offset += 18
    
    if mirror_view:
        cv2.putText(frame_copy, "Mirror View", 
                   (frame_copy.shape[1] - 100, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 105, 180), 1)
    
    return frame_copy

# Optimized Video Processor with frame skipping to prevent freezing
class VideoProcessor:
    def __init__(self):
        self.mirror_view = mirror_view
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.last_alert_time = 0
        self.last_auto_save_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_frame = None
        
    def recv(self, frame):
        self.frame_count += 1
        
        # Skip frames for better performance
        if self.frame_count % self.frame_skip != 0:
            # Return last processed frame or original frame
            if self.last_frame is not None:
                return av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")
            else:
                img = frame.to_ndarray(format="bgr24")
                if self.mirror_view:
                    img = cv2.flip(img, 1)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        img = frame.to_ndarray(format="bgr24")
        
        # Apply mirror if needed
        if self.mirror_view:
            img = cv2.flip(img, 1)
        
        # Calculate FPS periodically
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # Process frame with YOLO (simplified for speed)
        try:
            conf_threshold = 0.45
            results = model(img, conf=conf_threshold, iou=0.45, verbose=False, device='cpu', imgsz=320)
        except Exception as e:
            results = None
        
        current_detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                confidence = float(box.conf[0])
                
                if hasattr(box, 'xyxy'):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                else:
                    x1, y1, x2, y2 = box[0].tolist()
                
                current_detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence
                })
            
            # Update counts
            current_counts = defaultdict(int)
            for det in current_detections:
                current_counts[det['class']] += 1
            
            for obj, count in current_counts.items():
                st.session_state.object_counts[obj] = count
            
            # Reset counts for objects not detected
            for obj in list(st.session_state.object_counts.keys()):
                if obj not in current_counts:
                    st.session_state.object_counts[obj] = 0
            
            # Handle alerts
            current_time = time.time()
            if enable_alerts and (current_time - st.session_state.last_alert_time) >= 2:
                for det in current_detections:
                    if det['class'] in alert_objects:
                        st.session_state.detection_log.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'object': det['class'],
                            'confidence': f"{det['confidence']:.2f}"
                        })
                        # Keep only last 10 alerts
                        if len(st.session_state.detection_log) > 10:
                            st.session_state.detection_log = st.session_state.detection_log[-10:]
                        st.session_state.last_alert_time = current_time
                        break
        
        # Draw bounding boxes
        if current_detections:
            annotated_frame = draw_smooth_boxes(img, current_detections)
        else:
            annotated_frame = img
        
        # Add overlays
        final_frame = add_overlays(annotated_frame, st.session_state.object_counts, self.mirror_view)
        
        # Add FPS text
        cv2.putText(final_frame, f"FPS: {self.current_fps}", 
                   (8, final_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 105, 180), 1)
        
        # Handle frame saving (non-blocking)
        current_time = time.time()
        try:
            if save_frame_request and (current_time - st.session_state.last_save_time) > 1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"detected_frame_{timestamp}.jpg"
                filepath = os.path.join(SAVED_FRAMES_DIR, filename)
                cv2.imwrite(filepath, final_frame)
                st.session_state.last_save_time = current_time
                st.toast(f"📸 Frame saved!", icon="✨")
            
            if auto_save and (current_time - st.session_state.last_auto_save_time) >= 10:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"auto_saved_frame_{timestamp}.jpg"
                filepath = os.path.join(SAVED_FRAMES_DIR, filename)
                cv2.imwrite(filepath, final_frame)
                st.session_state.last_auto_save_time = current_time
        except:
            pass
        
        # Update placeholders
        try:
            if show_counting and st.session_state.object_counts:
                active_counts = {k: v for k, v in st.session_state.object_counts.items() if v > 0}
                if active_counts:
                    count_text = ""
                    for obj, count in active_counts.items():
                        count_text += f"**{obj}:** {count}  \n"
                    count_placeholder.markdown(count_text)
                else:
                    count_placeholder.write("No objects detected")
            
            if enable_alerts and st.session_state.detection_log:
                recent_alerts = st.session_state.detection_log[-3:]
                if recent_alerts:
                    alert_html = ""
                    for alert in recent_alerts:
                        alert_html += f"**{alert['object']}** detected\n\n"
                    alert_placeholder.warning(alert_html)
        except:
            pass
        
        # Store last frame
        self.last_frame = final_frame
        
        # Return processed frame
        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

# WebRTC streamer configuration with hidden buttons
if st.session_state.camera_active:
    with video_display_area.container():
        st.markdown("### 🎥 Live Camera Feed")
        
        # Parse resolution
        width, height = map(int, st.session_state.resolution.split('x'))
        
        # Configure WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": width, "max": width},
                    "height": {"ideal": height, "max": height},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            }
        )
        st.session_state.webrtc_ctx = webrtc_ctx
        
        # Show status message safely without accessing state.initialized
        if webrtc_ctx and webrtc_ctx.video_processor:
            st.success("✨ Camera active | Object detection running smoothly ✨")
        elif webrtc_ctx:
            st.info("🔄 Camera initializing... Please wait...")
        else:
            st.warning("⚠️ Camera not ready. Please check permissions.")
else:
    with video_display_area.container():
        st.info("🌸✨ Click 'Start Camera' below to begin object detection! ✨🌸\n\n💕 Make sure to allow camera permissions when prompted")
    
    if st.session_state.webrtc_ctx is not None:
        st.session_state.webrtc_ctx = None
