import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
from ultralytics import YOLO

# Cấu hình ICE mạnh hơn
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]}
    ]}
)

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

st.title("🐱 Cat Cam Live (Optimized)")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Ép kích thước ảnh cực nhỏ (160) để CPU server không bị nghẽn
    results = model.predict(img, conf=0.5, imgsz=160, verbose=False)
    
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Giao diện chính
ctx = webrtc_streamer(
    key="fixed-cam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320}, 
            "frameRate": {"ideal": 5}, # Chỉ chạy 5 hình/giây để giữ kết nối
            "facingMode": "environment",
        },
        "audio": False,
    },
    async_processing=True,
    # Thêm tham số này để giảm thiểu lỗi NoneType khi mất kết nối
    queued_video_frames_size=1, 
)
