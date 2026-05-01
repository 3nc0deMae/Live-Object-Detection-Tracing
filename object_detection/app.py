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
import threading

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
if 'resolution' not in st.session_state:
    st.session_state.resolution = "640x480"
if 'webrtc_ctx' not in st.session_state:
    st.session_state.webrtc_ctx = None
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'mirror_view_enabled' not in st.session_state:
    st.session_state.mirror_view_enabled = True
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Thread-safe storage for shared data
class SharedData:
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.detection_log = []
        self.last_alert_time = 0
        self.last_auto_save_time = 0
        self.last_save_time = 0
        self.mirror_view_enabled = True
        self.enable_alerts = True
        self.alert_objects = ["person"]
        self.auto_save = False
        self.show_counting = True
        self.save_request = False
        self.lock = threading.Lock()
    
    def update_counts(self, counts):
        with self.lock:
            self.object_counts = counts
    
    def get_counts(self):
        with self.lock:
            return self.object_counts.copy()
    
    def add_alert(self, alert):
        with self.lock:
            self.detection_log.append(alert)
            if len(self.detection_log) > 10:
                self.detection_log = self.detection_log[-10:]
    
    def get_alerts(self):
        with self.lock:
            return self.detection_log.copy()
    
    def set_mirror(self, value):
        with self.lock:
            self.mirror_view_enabled = value
    
    def get_mirror(self):
        with self.lock:
            return self.mirror_view_enabled
    
    def set_enable_alerts(self, value):
        with self.lock:
            self.enable_alerts = value
    
    def get_enable_alerts(self):
        with self.lock:
            return self.enable_alerts
    
    def set_alert_objects(self, value):
        with self.lock:
            self.alert_objects = list(value)
    
    def get_alert_objects(self):
        with self.lock:
            return list(self.alert_objects)
    
    def set_auto_save(self, value):
        with self.lock:
            self.auto_save = value
    
    def get_auto_save(self):
        with self.lock:
            return self.auto_save
    
    def set_show_counting(self, value):
        with self.lock:
            self.show_counting = value
    
    def get_show_counting(self):
        with self.lock:
            return self.show_counting
    
    def set_save_request(self, value):
        with self.lock:
            self.save_request = value
    
    def get_save_request(self):
        with self.lock:
            return self.save_request

# Initialize shared data as module-level
if 'shared_data' not in globals():
    shared_data = SharedData()

# Cache the model (no Streamlit UI calls inside - they persist on reruns)
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_input, verbose=False)
        return model
    except Exception:
        return None

model = load_model()
if model is not None:
    st.session_state.model_ready = True


# Object colors
OBJECT_COLORS = {
    'person': (255, 105, 180),
    'cell phone': (135, 206, 235),
    'laptop': (100, 149, 237),
    'tv': (70, 130, 180),
    'bottle': (144, 238, 144),
    'book': (255, 182, 193),
    'cat': (255, 215, 0),
    'dog': (255, 140, 0),
    'bird': (255, 165, 0),
    'chair': (221, 160, 221),
    'car': (255, 99, 71),
    'bus': (255, 20, 147),
    'default': (147, 112, 219)
}

def get_object_color(class_name):
    return OBJECT_COLORS.get(class_name.lower(), OBJECT_COLORS['default'])

def draw_boxes(frame, boxes_data):
    for box_data in boxes_data:
        box = box_data['box']
        class_name = box_data['class']
        confidence = box_data['confidence']
        
        x1, y1, x2, y2 = map(int, box)
        color = get_object_color(class_name)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f"{class_name.upper()} {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color, -1
        )
        cv2.putText(
            frame, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return frame

def add_overlays(frame, object_counts, mirror_view_enabled, show_counting):
    """Added show_counting as parameter to avoid relying on outer scope."""
    if show_counting and object_counts:
        active_counts = {k: v for k, v in object_counts.items() if v > 0}
        if active_counts:
            y_offset = 30
            for obj, count in list(active_counts.items())[:6]:
                cv2.putText(
                    frame, f"{obj}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )
                y_offset += 25
    
    if mirror_view_enabled:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (frame.shape[1] - 170, 5),
            (frame.shape[1] - 5, 35),
            (255, 105, 180), -1
        )
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        cv2.putText(
            frame, "MIRROR MODE",
            (frame.shape[1] - 165, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
    
    return frame


class VideoProcessor:
    INFERENCE_SIZE = 416  # Downscale for YOLO inference (320->416 for better accuracy)
    PROCESS_EVERY_N = 4   # Run detection every N frames (3->4 for less lag)

    def __init__(self):
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.frames_in_second = 0
        self.current_fps = 0
        self.model = model
        self.model_available = model is not None
        self.last_detections = []
        self.shared_data = shared_data
        self._save_thread = None
        
    def _save_frame_async(self, filepath, img):
        """Save frame to disk in a background thread."""
        try:
            cv2.imwrite(filepath, img)
        except Exception:
            pass

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            orig_h, orig_w = img.shape[:2]

            # Cache all shared_data reads once per frame (reduces lock contention)
            sd = self.shared_data
            mirror_enabled = sd.get_mirror()
            show_counting = sd.get_show_counting()
            enable_alerts = sd.get_enable_alerts()
            alert_objects = sd.get_alert_objects()
            auto_save = sd.get_auto_save()
            save_requested = sd.get_save_request()

            # FPS counter
            self.frames_in_second += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.frames_in_second
                self.frames_in_second = 0
                self.fps_start_time = current_time

            # Run detection every N frames
            self.frame_count += 1
            if (self.frame_count % self.PROCESS_EVERY_N == 0
                    and self.model_available and self.model is not None):
                try:
                    # Downscale for faster inference
                    inference_img = cv2.resize(
                        img, (self.INFERENCE_SIZE, self.INFERENCE_SIZE)
                    )
                    results = self.model(
                        inference_img, conf=0.5, iou=0.45,
                        imgsz=self.INFERENCE_SIZE, verbose=False
                    )

                    if (results and len(results) > 0
                            and results[0].boxes is not None):
                        boxes = results[0].boxes
                        names = results[0].names

                        # Scale factor to map inference coords back to original
                        scale_x = orig_w / self.INFERENCE_SIZE
                        scale_y = orig_h / self.INFERENCE_SIZE

                        self.last_detections = []
                        current_counts = defaultdict(int)

                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            # Scale boxes to original frame size
                            x1 *= scale_x; y1 *= scale_y
                            x2 *= scale_x; y2 *= scale_y
                            class_id = int(box.cls[0])
                            class_name = names[class_id]
                            confidence = float(box.conf[0])

                            if confidence > 0.5:
                                self.last_detections.append({
                                    'box': [x1, y1, x2, y2],
                                    'class': class_name,
                                    'confidence': confidence
                                })
                                current_counts[class_name] += 1

                        sd.update_counts(current_counts)

                        # Handle alerts
                        time_since_alert = (
                            current_time - sd.last_alert_time
                        )
                        if enable_alerts and time_since_alert >= 2:
                            for det in self.last_detections:
                                if det['class'] in alert_objects:
                                    sd.add_alert({
                                        'timestamp': datetime.now().strftime(
                                            "%H:%M:%S"
                                        ),
                                        'object': det['class'],
                                        'confidence': f"{det['confidence']:.2f}"
                                    })
                                    sd.last_alert_time = current_time
                                    break
                except Exception:
                    pass

            # Apply mirror BEFORE drawing boxes so text stays readable
            if mirror_enabled:
                img = cv2.flip(img, 1)

            # Draw boxes on frame (after flip so labels are correct)
            if self.last_detections:
                if mirror_enabled:
                    mirrored_detections = []
                    for det in self.last_detections:
                        x1, y1, x2, y2 = det['box']
                        mirrored_detections.append({
                            'box': [orig_w - x2, y1, orig_w - x1, y2],
                            'class': det['class'],
                            'confidence': det['confidence']
                        })
                    img = draw_boxes(img, mirrored_detections)
                else:
                    img = draw_boxes(img, self.last_detections)

            # Add overlays
            current_counts = sd.get_counts()
            img = add_overlays(img, current_counts, mirror_enabled, show_counting)

            # Add FPS counter
            cv2.putText(
                img, f"{self.current_fps} FPS",
                (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2
            )

            # Handle manual save (offload disk I/O)
            if save_requested and (current_time - sd.last_save_time) > 1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filepath = os.path.join(
                    SAVED_FRAMES_DIR, f"detected_frame_{timestamp}.jpg"
                )
                t = threading.Thread(
                    target=self._save_frame_async,
                    args=(filepath, img.copy()),
                    daemon=True
                )
                t.start()
                sd.last_save_time = current_time
                sd.set_save_request(False)

            # Handle auto-save (offload disk I/O)
            if auto_save and (current_time - sd.last_auto_save_time) >= 10:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filepath = os.path.join(
                    SAVED_FRAMES_DIR, f"auto_saved_frame_{timestamp}.jpg"
                )
                t = threading.Thread(
                    target=self._save_frame_async,
                    args=(filepath, img.copy()),
                    daemon=True
                )
                t.start()
                sd.last_auto_save_time = current_time

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank, "Camera Active", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            return av.VideoFrame.from_ndarray(blank, format="bgr24")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Dancing+Script:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');

    /* ═══════════════════════════════════════════
       DESIGN TOKENS
       ═══════════════════════════════════════════ */
    :root {
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.4);
        --glass-hover: rgba(255, 255, 255, 0.45);
        --accent-pink: #ff69b4;
        --accent-purple: #9370db;
        --accent-blue: #4169e1;
        --text-primary: #4a2040;
        --text-secondary: #7b4f7b;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.12);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.16);
        --shadow-glow: 0 0 20px rgba(255, 105, 180, 0.25);
        --radius-sm: 12px;
        --radius-md: 16px;
        --radius-lg: 24px;
        --radius-full: 50px;
        --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-smooth: 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-bounce: 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    /* ═══════════════════════════════════════════
       BASE / APP
       ═══════════════════════════════════════════ */
    .stApp {
        background: linear-gradient(135deg, #ffe6f0 0%, #e6d5ff 50%, #d5e8ff 100%);
        padding-top: 56px !important;
        padding-bottom: 52px !important;
    }

    html, body, .stMarkdown, .stText, .stButton,
    .stCheckbox, .stSelectbox, .stSlider {
        font-family: 'Quicksand', sans-serif !important;
    }

    /* ═══════════════════════════════════════════
       RIBBONS
       ═══════════════════════════════════════════ */
    .top-ribbon {
        position: fixed;
        top: 0; left: 0; right: 0;
        text-align: center;
        font-size: 22px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 10px;
        color: white;
        font-family: 'Dancing Script', cursive;
        letter-spacing: 4px;
        z-index: 999;
        box-shadow: var(--shadow-md);
        backdrop-filter: blur(10px);
    }

    .bottom-ribbon {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        text-align: center;
        font-size: 16px;
        background: linear-gradient(90deg, #ffb6c1, #dda0dd, #87ceeb);
        padding: 8px;
        color: white;
        font-family: 'Dancing Script', cursive;
        letter-spacing: 4px;
        z-index: 999;
        backdrop-filter: blur(10px);
    }

    /* ═══════════════════════════════════════════
       TYPOGRAPHY
       ═══════════════════════════════════════════ */
    h1, h2, h3, .stSubheader {
        font-family: 'Dancing Script', cursive !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, var(--accent-pink), var(--accent-purple), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h1 {
        font-size: 3.2em !important;
        margin-top: 0 !important;
        text-align: center !important;
        letter-spacing: 1px;
    }

    h3 {
        letter-spacing: 0.5px;
    }

    /* ═══════════════════════════════════════════
       BUTTONS — Glassmorphism + Hover
       ═══════════════════════════════════════════ */
    .stButton > button {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(14px) !important;
        border: 1.5px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px;
        transition: all var(--transition-smooth) !important;
        border-radius: var(--radius-full) !important;
        padding: 4px 28px !important;
        min-height: 36px !important;
        box-shadow: var(--shadow-sm) !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        background: var(--glass-hover) !important;
        border-color: var(--accent-pink) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: var(--shadow-glow) !important;
        color: var(--text-primary) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: var(--shadow-sm) !important;
        transition-duration: 0.1s !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(255,182,193,0.6), rgba(221,160,221,0.6), rgba(135,206,235,0.6)) !important;
        border: 1.5px solid rgba(255,182,193,0.7) !important;
        color: white !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.15);
        font-size: 17px !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(255,182,193,0.8), rgba(221,160,221,0.8), rgba(135,206,235,0.8)) !important;
        box-shadow: 0 8px 28px rgba(255,105,180,0.35) !important;
        transform: translateY(-3px) scale(1.03) !important;
    }

    /* ═══════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(255, 240, 245, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 0 var(--radius-lg) var(--radius-lg) 0 !important;
        border-right: 1.5px solid rgba(255,182,193,0.35) !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.06) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 20px 16px 20px 16px !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div.stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] a {
        font-family: 'Quicksand', sans-serif !important;
    }

    /* Fix sidebar icon rendering */
    [data-testid="stSidebar"] .material-icons,
    [data-testid="stSidebar"] .material-icons-outlined,
    [data-testid="stSidebar"] [data-testid="stIconFont"],
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebar"] button[kind="header"] *,
    [data-testid="stSidebar"] button[aria-label] span:not(.stMarkdown) {
        font-family: 'Material Icons Outlined', 'Material Icons', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar checkbox hover */
    [data-testid="stSidebar"] .stCheckbox:hover {
        background: rgba(255,255,255,0.3);
        border-radius: var(--radius-sm);
        transition: background var(--transition-fast);
    }

    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        transition: all var(--transition-fast) !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: var(--accent-pink) !important;
        box-shadow: var(--shadow-glow) !important;
    }

    /* ═══════════════════════════════════════════
       CONTAINERS / CARDS
       ═══════════════════════════════════════════ */
    .stContainer {
        border-radius: var(--radius-md) !important;
        transition: all var(--transition-smooth) !important;
    }

    [data-testid="stVerticalBlock"] > [style*="border"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all var(--transition-smooth) !important;
    }

    [data-testid="stVerticalBlock"] > [style*="border"]:hover {
        box-shadow: var(--shadow-md) !important;
        border-color: rgba(255,182,193,0.6) !important;
        transform: translateY(-1px);
    }

    /* ═══════════════════════════════════════════
       ALERTS / MESSAGES
       ═══════════════════════════════════════════ */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: var(--radius-sm) !important;
        border-left: 4px solid var(--accent-pink) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all var(--transition-smooth) !important;
    }

    .stSuccess:hover, .stInfo:hover, .stWarning:hover, .stError:hover {
        box-shadow: var(--shadow-md) !important;
        transform: translateX(2px);
    }

    /* ═══════════════════════════════════════════
       EXPANDERS
       ═══════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid var(--glass-border) !important;
        transition: all var(--transition-smooth) !important;
    }

    .streamlit-expanderHeader:hover {
        background: var(--glass-hover) !important;
        border-color: var(--accent-pink) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ═══════════════════════════════════════════
       DOWNLOAD BUTTONS
       ═══════════════════════════════════════════ */
    .stDownloadButton > button {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1.5px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
        transition: all var(--transition-smooth) !important;
    }

    .stDownloadButton > button:hover {
        background: var(--glass-hover) !important;
        border-color: var(--accent-purple) !important;
        box-shadow: 0 4px 16px rgba(147,112,219,0.25) !important;
        transform: translateY(-1px) !important;
    }

    /* ═══════════════════════════════════════════
       CHECKBOXES
       ═══════════════════════════════════════════ */
    .stCheckbox > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        transition: all var(--transition-fast) !important;
    }

    .stCheckbox > label:hover {
        color: var(--accent-pink) !important;
    }

    /* ═══════════════════════════════════════════
       WEBRTC VIDEO
       ═══════════════════════════════════════════ */
    .streamlit-webrtc video {
        border-radius: var(--radius-lg) !important;
        width: 100% !important;
        height: auto !important;
        background: rgba(0,0,0,0.05) !important;
        box-shadow: var(--shadow-md) !important;
        transition: box-shadow var(--transition-smooth) !important;
    }

    .streamlit-webrtc video:hover {
        box-shadow: var(--shadow-lg) !important;
    }

    .streamlit-webrtc .stButton,
    .streamlit-webrtc button {
        display: block !important;
    }

    /* ═══════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255,230,240,0.4);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ffb6c1, #dda0dd, #87ceeb);
        border-radius: 10px;
        transition: all var(--transition-fast);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff69b4, #9370db, #4169e1);
        box-shadow: 0 0 6px rgba(255,105,180,0.4);
    }

    /* ═══════════════════════════════════════════
       DIVIDER
       ═══════════════════════════════════════════ */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(255,182,193,0.5), rgba(221,160,221,0.5), rgba(135,206,235,0.5), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ═══════════════════════════════════════════
       ANIMATIONS
       ═══════════════════════════════════════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .stApp > header + div > div {
        animation: fadeInUp 0.5s ease-out;
    }
</style>

<div class="top-ribbon">
    ✨🌸✨ Object Detection ✨🌸✨
</div>
<div class="bottom-ribbon">
    ✨ Made with 💕 by AI Magic | Object Detection & Tracking ✨
</div>
""", unsafe_allow_html=True)


st.markdown(
    "<h1>✨ Live Object Detection & Tracing ✨</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#6b3e6b; font-size:1.2em;"
    " font-family:Quicksand; margin-bottom:30px;'>"
    "🌸 Point your camera at objects to identify them in real-time with AI magic 🌸</p>",
    unsafe_allow_html=True
)

# Sidebar controls
with st.sidebar:
    st.markdown("### ✨💖 Settings 💖✨")
    
    st.markdown("#### 🎥 Camera Settings")
    mirror_view = st.checkbox(
        "🪞 Mirror View (Inverted)",
        value=st.session_state.mirror_view_enabled
    )
    if mirror_view != st.session_state.mirror_view_enabled:
        st.session_state.mirror_view_enabled = mirror_view
        shared_data.set_mirror(mirror_view)
    
    st.markdown("#### 📱 Quality & Resolution")
    resolution_options = {
        "Low (480p) - Fastest": "640x480",
        "Medium (720p) - Balanced": "1280x720",
    }
    selected_resolution = st.selectbox(
        "🎬 Select Resolution",
        options=list(resolution_options.keys()),
        index=0,
        help="Higher resolution = better quality but more processing power"
    )
    st.session_state.resolution = resolution_options[selected_resolution]
    
    st.markdown("#### 📊 Object Counting")
    show_counting = st.checkbox(
        "🔢 Show Object Counting",
        value=shared_data.get_show_counting()
    )
    shared_data.set_show_counting(show_counting)
    
    st.markdown("#### 🚨 Alert System")
    enable_alerts = st.checkbox(
        "🔔 Enable Alerts",
        value=shared_data.get_enable_alerts()
    )
    shared_data.set_enable_alerts(enable_alerts)
    
    alert_objects = st.multiselect(
        "🎯 Alert for these objects",
        options=[
            "person", "cell phone", "bottle", "laptop",
            "chair", "book", "tv", "cat", "dog", "bird"
        ],
        default=shared_data.get_alert_objects()
    )
    shared_data.set_alert_objects(alert_objects)
    
    st.markdown("#### 💾 Frame Saving")
    if st.button("📸 Save Current Frame", use_container_width=True):
        shared_data.set_save_request(True)
    
    auto_save = st.checkbox(
        "🤖 Auto-save every 10 seconds",
        value=shared_data.get_auto_save()
    )
    shared_data.set_auto_save(auto_save)
    
    if st.button("🔄 Reset All Counters", use_container_width=True):
        st.session_state.object_counts.clear()
        st.session_state.detection_log.clear()
        shared_data.__init__()
        st.success("✨ Counters reset successfully! ✨")
    
    st.markdown("---")
    st.markdown("#### 🗑️ Manage Saved Frames")
    if st.button("🗑️ Delete ALL Saved Frames", use_container_width=True):
        saved_frames_list = [
            f for f in os.listdir(SAVED_FRAMES_DIR)
            if f.startswith(("detected_frame_", "auto_saved_frame_"))
            and f.endswith(".jpg")
        ]
        if saved_frames_list:
            for frame_file in saved_frames_list:
                try:
                    os.remove(os.path.join(SAVED_FRAMES_DIR, frame_file))
                except Exception:
                    pass
            st.success(f"✅ Deleted {len(saved_frames_list)} saved frames 💕")
            st.rerun()
        else:
            st.info("💝 No saved frames to delete")

# Camera feed
if st.session_state.camera_active:
    st.markdown("### 🎥 Live Camera Feed")
    
    if st.session_state.mirror_view_enabled:
        st.caption("🪞 Mirror Mode: ON - Image is flipped horizontally")
    else:
        st.caption("🎥 Normal Mode")
    
    width, height = map(int, st.session_state.resolution.split('x'))
    
    if not st.session_state.model_ready or model is None:
        st.warning("⚠️ AI Model is still loading...")
    else:
        st.success("✅ AI Model Ready - Detecting objects in real-time!")
    
    # VideoProcessor is now defined ABOVE this call ✅
    webrtc_ctx = webrtc_streamer(
        key="working-object-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": width, "max": width},
                "height": {"ideal": height, "max": height},
                "frameRate": {"ideal": 20, "max": 25},
            },
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:443?transport=tcp",
                        "turn:openrelay.metered.ca:443?transport=udp",
                        "turn:openrelay.metered.ca:80?transport=tcp",
                        "turn:openrelay.metered.ca:80?transport=udp",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        },
    )
    st.session_state.webrtc_ctx = webrtc_ctx
    
    if webrtc_ctx and webrtc_ctx.video_processor:
        st.success("✨ Camera Active | Real-time Detection Running ✨")
    
    # Stop camera button below the feed
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div style="visibility:hidden">Stop Camera</div>', unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            if st.button(
                "⏹️ Stop Camera",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.camera_active = False
                st.session_state.webrtc_ctx = None
                st.rerun()
    with col3:
        with st.container(border=True):
            st.markdown('<div style="visibility:hidden">Stop Camera</div>', unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='"
        "border-left: 5px solid #ff69b4; "
        "padding: 16px 20px; "
        "margin: 10px 0; "
        "background: rgba(255,255,255,0.25); "
        "backdrop-filter: blur(12px); "
        "border-radius: 0 12px 12px 0; "
        "font-family: Quicksand, sans-serif; "
        "color: #4a2040; "
        "line-height: 1.8; "
        "box-shadow: 0 2px 8px rgba(0,0,0,0.08);"
        "'>"
        "🌸✨ Click 'Start Camera' to begin real-time object detection! ✨🌸<br/><br/>"
        "💕 Make sure to allow camera permissions<br/><br/>"
        "🎯 Detects: people, phones, laptops, bottles, books, pets, and more!"
        "</div>",
        unsafe_allow_html=True
    )
    if st.session_state.webrtc_ctx is not None:
        st.session_state.webrtc_ctx = None
    # Start camera button below the info message
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div style="visibility:hidden">Start Camera</div>', unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            if st.button(
                "📷 Start Camera",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.camera_active = True
                st.session_state.object_counts.clear()
                st.session_state.detection_log.clear()
                st.rerun()
    with col3:
        with st.container(border=True):
            st.markdown('<div style="visibility:hidden">Start Camera</div>', unsafe_allow_html=True)

# Display area
st.markdown("---")
disp_col1, disp_col2, disp_col3 = st.columns(3)

with disp_col1:
    with st.container(border=True):
        st.markdown("#### 📊 Object Count")
        count_placeholder = st.empty()

with disp_col2:
    with st.container(border=True):
        st.markdown("#### 🚨 Recent Alerts")
        alert_placeholder = st.empty()

with disp_col3:
    with st.container(border=True):
        st.markdown("#### 💾 Saved Frames")
        saved_frames = [
            f for f in os.listdir(SAVED_FRAMES_DIR)
            if f.startswith(("detected_frame_", "auto_saved_frame_"))
            and f.endswith(".jpg")
        ]
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
                            if st.button(
                                "🗑️ Delete",
                                key=f"delete_{frame_file}_{idx}"
                            ):
                                try:
                                    os.remove(frame_path)
                                    st.success(f"✅ Deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
        else:
            st.write("💝 No frames saved yet")

# Update UI from shared data
if (st.session_state.camera_active
        and st.session_state.webrtc_ctx
        and st.session_state.webrtc_ctx.video_processor):
    
    current_counts = shared_data.get_counts()
    if show_counting:
        active_counts = {k: v for k, v in current_counts.items() if v > 0}
        if active_counts:
            count_text = ""
            for obj, count in active_counts.items():
                count_text += f"**{obj}:** {count}  \n"
            count_placeholder.markdown(count_text)
        else:
            count_placeholder.write("No objects detected")
    
    if enable_alerts:
        alerts = shared_data.get_alerts()
        if alerts:
            recent_alerts = alerts[-3:]
            alert_html = ""
            for alert in recent_alerts:
                alert_html += (
                    f"🔔 **{alert['object']}** detected "
                    f"({alert['confidence']})\n\n"
                )
            alert_placeholder.warning(alert_html)