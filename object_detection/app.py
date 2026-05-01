import streamlit as st
from ultralytics import YOLO
import cv2
from collections import defaultdict
from datetime import datetime
import os
import time
import numpy as np

# Load external CSS
def load_css():
    with open('styles.css', 'r') as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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

# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Load CSS styles
load_css()

# HTML for ribbons
st.markdown("""
<div class="top-ribbon">
    ✨🌸✨ Object Detection ✨🌸✨
</div>

<div class="bottom-ribbon">
    ✨ Made with 💕 by AI Magic | Object Detection & Tracking ✨
</div>
""", unsafe_allow_html=True)

# Title with ribbon style
st.markdown("<h1>✨ Live Object Detection & Tracing ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b3e6b; font-size: 1.2em; font-family: Quicksand; margin-bottom: 30px;'>🌸 Point your camera at objects to identify them in real-time with AI magic 🌸</p>", unsafe_allow_html=True)

# Sidebar for add-ons controls
with st.sidebar:
    st.markdown("### ✨💖 Settings 💖✨")
    
    # Camera settings
    st.markdown("#### 🎥 Camera Settings")
    mirror_view = st.checkbox("🪞 Mirror View (Inverted)", value=True)
    
    # Quality Resolution Feature
    st.markdown("#### 📱 Quality & Resolution")
    resolution_options = {
        "Low (480p) - Fastest": "640x480",
        "Medium (720p) - Balanced": "1280x720", 
        "High (1080p) - High Quality": "1920x1080",
        "Ultra HD (4K) - Best Quality": "3840x2160"
    }
    selected_resolution = st.selectbox(
        "🎬 Select Resolution",
        options=list(resolution_options.keys()),
        index=0,
        help="Higher resolution = better quality but more processing power needed"
    )
    st.session_state.resolution = resolution_options[selected_resolution]
    
    # Show resolution info
    if "3840" in st.session_state.resolution:
        st.info("🌟 4K Ultra HD mode - Best quality, high performance needed")
    elif "1920" in st.session_state.resolution:
        st.info("💕 1080p High Quality - Great detail, good performance")
    elif "1280" in st.session_state.resolution:
        st.info("✨ 720p Balanced - Good quality and speed")
    else:
        st.success("🌸 480p Fast mode - Optimized for smooth performance")
    
    # Object counting
    st.markdown("#### 📊 Object Counting")
    show_counting = st.checkbox("🔢 Show Object Counting", value=True)
    
    # Alert system
    st.markdown("#### 🚨 Alert System")
    enable_alerts = st.checkbox("🔔 Enable Alerts", value=True)
    alert_objects = st.multiselect(
        "🎯 Alert for these objects",
        options=["person", "cell phone", "bottle", "laptop", "chair", "book", "tv", "cat", "dog", "bird"],
        default=["person"]
    )
    
    # Frame saving
    st.markdown("#### 💾 Frame Saving")
    save_frame_request = st.button("📸 Save Current Frame", use_container_width=True)
    auto_save = st.checkbox("🤖 Auto-save every 10 seconds", value=False)
    
    # Reset counter
    if st.button("🔄 Reset All Counters", use_container_width=True):
        st.session_state.object_counts.clear()
        st.session_state.detection_log.clear()
        st.success("✨ Counters reset successfully! ✨")
    
    # Delete all saved frames
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

# Create placeholder for video feed
video_placeholder = st.empty()

# Center the camera controls below the video feed
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
            st.rerun()

# Display add-ons information below the camera
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

# Define distinct colors for different object classes
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
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        
        if confidence:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = class_name
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame_copy, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_copy

def add_overlays(frame, object_counts, mirror_view):
    frame_copy = frame.copy()
    
    if show_counting and object_counts:
        active_counts = {k: v for k, v in object_counts.items() if v > 0}
        if active_counts:
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (5, 5), (220, 40 + len(active_counts) * 25), (255, 182, 193), -1)
            frame_copy = cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0)
            
            y_offset = 30
            for obj, count in list(active_counts.items())[:5]:
                cv2.putText(frame_copy, f"{obj}: {count}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.55, (255, 255, 255), 2)
                y_offset += 25
    
    if mirror_view:
        cv2.putText(frame_copy, "Mirror View", 
                   (frame_copy.shape[1] - 130, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2)
    
    res_display = st.session_state.resolution.replace('x', ' x ')
    cv2.putText(frame_copy, f"Resolution: {res_display}", 
               (frame_copy.shape[1] - 220, frame_copy.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2)
    
    return frame_copy

def run_camera():
    width, height = map(int, st.session_state.resolution.split('x'))
    
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        st.error("Cannot access camera. Please check your camera connection.")
        st.session_state.camera_active = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.info(f"Camera running at: {actual_width}x{actual_height} resolution")
    
    frame_count = 0
    last_alert_time = 0
    last_auto_save_time = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    while st.session_state.camera_active:
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.1)
            continue
        
        if mirror_view:
            frame = cv2.flip(frame, 1)
        
        frame_count += 1
        
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        conf_threshold = 0.4 if "3840" in st.session_state.resolution or "1920" in st.session_state.resolution else 0.5
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=0.5,
            verbose=False,
            device='cpu'
        )
        
        current_detections = []
        if results[0].boxes is not None:
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
            
            current_counts = defaultdict(int)
            for det in current_detections:
                current_counts[det['class']] += 1
            
            for obj, count in current_counts.items():
                st.session_state.object_counts[obj] = count
            
            for obj in list(st.session_state.object_counts.keys()):
                if obj not in current_counts:
                    st.session_state.object_counts[obj] = 0
            
            current_time = time.time()
            if enable_alerts and (current_time - last_alert_time) >= 2:
                for det in current_detections:
                    if det['class'] in alert_objects:
                        st.session_state.detection_log.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'object': det['class'],
                            'confidence': f"{det['confidence']:.2f}"
                        })
                        last_alert_time = current_time
                        break
        
        if current_detections:
            annotated_frame = draw_smooth_boxes(frame, current_detections)
        else:
            annotated_frame = frame
        
        final_frame = add_overlays(annotated_frame, st.session_state.object_counts, mirror_view)
        
        cv2.putText(final_frame, f"FPS: {current_fps}", 
                   (10, final_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2)
        
        current_time = time.time()
        if save_frame_request and (current_time - st.session_state.last_save_time) > 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"detected_frame_{timestamp}.jpg"
            filepath = os.path.join(SAVED_FRAMES_DIR, filename)
            
            cv2.imwrite(filepath, final_frame)
            
            st.session_state.last_save_time = current_time
            st.success(f"Frame saved: {filename}")
            time.sleep(0.5)
            st.rerun()
        
        if auto_save and (current_time - last_auto_save_time) >= 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"auto_saved_frame_{timestamp}.jpg"
            filepath = os.path.join(SAVED_FRAMES_DIR, filename)
            
            cv2.imwrite(filepath, final_frame)
            
            last_auto_save_time = current_time
            st.info(f"Auto-saved: {filename}")
        
        display_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
        
        if show_counting and st.session_state.object_counts:
            active_counts = {k: v for k, v in st.session_state.object_counts.items() if v > 0}
            if active_counts:
                count_text = ""
                for obj, count in active_counts.items():
                    count_text += f"**{obj}:** {count}  \n"
                count_placeholder.markdown(count_text)
            else:
                count_placeholder.write("No objects detected")
        else:
            count_placeholder.write("Object counting disabled")
        
        if enable_alerts and st.session_state.detection_log:
            recent_alerts = st.session_state.detection_log[-3:]
            if recent_alerts:
                alert_html = ""
                for alert in recent_alerts:
                    alert_html += f"**{alert['object']}** detected ({alert['confidence']})\n\n"
                alert_placeholder.warning(alert_html)
            else:
                alert_placeholder.write("No recent alerts")
        else:
            alert_placeholder.write("Alerts disabled")
        
        delay = 0.01 if "480" in st.session_state.resolution else 0.02 if "720" in st.session_state.resolution else 0.03
        time.sleep(delay)
    
    cap.release()
    cv2.destroyAllWindows()

if st.session_state.camera_active:
    try:
        run_camera()
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        st.session_state.camera_active = False
        st.rerun()
else:
    video_placeholder.info("Click 'Start Camera' below to begin object detection\n\nMake sure to allow camera permissions when prompted\n\nTip: Adjust resolution in sidebar for best quality/performance balance")