import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import torch # Thêm thư viện torch để kiểm tra GPU
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Cấu hình WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. Giao diện Sidebar - Chọn thiết bị xử lý
st.sidebar.title("Cấu hình hệ thống")

# Kiểm tra xem máy tính có GPU (CUDA) không
cuda_available = torch.cuda.is_available()
device_options = ["CPU"]
if cuda_available:
    device_options.append("GPU (CUDA)")
    device_default_index = 1 # Ưu tiên chọn GPU nếu có
else:
    device_default_index = 0

device_choice = st.sidebar.radio(
    "Thiết bị xử lý (Inference Device):",
    device_options,
    index=device_default_index,
    help="GPU sẽ cho tốc độ nhanh hơn nhiều so với CPU."
)

# Quyết định chọn device cho YOLO
# 'cpu' hoặc 0 (thường là GPU đầu tiên)
target_device = 'cpu' if device_choice == "CPU" else 0

# 3. Tải mô hình YOLO (truyền tham số device vào)
@st.cache_resource
def load_model(device):
    model = YOLO('yolov8n.pt') 
    # Chuyển model sang thiết bị đã chọn
    model.to(device)
    return model

model = load_model(target_device)

# Hiển thị thông tin thiết bị đang dùng
if target_device == 0:
    st.sidebar.success(f"🚀 Đang sử dụng: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("🐢 Đang sử dụng: CPU")

# --- PHẦN GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center;'>🐱 AI Cat Detector (CPU/GPU)</h1>", unsafe_allow_html=True)

mode = st.sidebar.selectbox(
    "Chế độ sử dụng",
    ["🎥 Webcam Thời gian thực", "📸 Chụp ảnh & Tải file"]
)

# --- CHỨC NĂNG 1: WEBCAM THỜI GIAN THỰC ---
if mode == "🎥 Webcam Thời gian thực":
    st.subheader("Real-time Detection")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Chạy nhận diện với device đã chọn
        # Chúng ta dùng trực tiếp target_device ở đây
        results = model.predict(img, conf=0.4, device=target_device, verbose=False)
        
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
    key="cat-detector",
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320, "max": 640}, # Giảm độ phân giải xuống 320p cho nhẹ
            "frameRate": {"ideal": 10, "max": 15}, # Giảm FPS xuống để tránh lag
            "facingMode": "environment",
        },
        "audio": False,
    },
    async_processing=True,
)

# --- CHỨC NĂNG 2: CHỤP ẢNH & TẢI FILE ---
else:
    st.subheader("Chụp ảnh hoặc Tải file")
    choice = st.radio("Nguồn ảnh:", ["Camera", "Tải ảnh từ máy"], horizontal=True)
    
    img_input = None
    if choice == "Camera":
        img_input = st.camera_input("Chụp một bức ảnh")
    else:
        img_input = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

    if img_input:
        image = Image.open(img_input)
        img_array = np.array(image)
        
        with st.spinner(f"Đang xử lý trên {device_choice}..."):
            # Chạy nhận diện
            results = model.predict(img_array, conf=0.4, device=target_device)
            st.image(results[0].plot(), use_container_width=True)
            
            count = len(results[0].boxes)
            st.success(f"Phát hiện {count} đối tượng!")

# 4. CSS Tùy chỉnh
st.markdown("""
    <style>
    video { border-radius: 15px; border: 2px solid #ff4b4b; }
    .stSidebar { background-color: #f8f9fa; }
    </style>

    """, unsafe_allow_html=True)
