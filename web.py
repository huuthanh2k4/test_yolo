import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
from ultralytics import YOLO

# Cấu hình ICE: Thêm nhiều server để tăng tỉ lệ kết nối trên điện thoại
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]}
    ]}
)

@st.cache_resource
def load_model():
    # Sử dụng bản nano để CPU Cloud xử lý kịp
    return YOLO('yolov8n.pt') 

model = load_model()

st.title("🐱 Cat Detector Live")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Ép YOLO chạy ở kích thước nhỏ (320) để tránh treo CPU server
    results = model.predict(img, conf=0.4, imgsz=320, verbose=False)
    
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Giao diện chính - Đã xóa tham số gây lỗi 'queued_video_frames_size'
webrtc_streamer(
    key="cat-cam-fixed",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480}, 
            "frameRate": {"ideal": 10},
            "facingMode": "environment", # Ưu tiên camera sau của Xiaomi 11T
        },
        "audio": False,
    },
    async_processing=True,
)
