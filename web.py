import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
from ultralytics import YOLO

# Cấu hình ICE: Chỉ dùng 1 STUN duy nhất của Google để tránh xung đột handshake
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_model():
    # Load bản Nano để nhẹ nhất có thể cho CPU của Streamlit Cloud
    return YOLO('yolov8n.pt') 

model = load_model()

st.title("🐱 Cat Cam Live (Xiaomi 11T)")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Giảm độ tự tin xuống một chút và giới hạn số lượng đối tượng để CPU xử lý kịp
    results = model.predict(img, conf=0.5, iou=0.45, verbose=False, imgsz=320)
    
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Giao diện chính
st.info("Lưu ý: Nếu màn hình đen, hãy thử chuyển từ 4G sang Wi-Fi hoặc ngược lại.")

ctx = webrtc_streamer(
    key="cat-detector-ultra",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480, "max": 480}, # Giảm độ phân giải xuống mức thấp hơn
            "frameRate": {"ideal": 7, "max": 10}, # Giảm FPS xuống để CPU server xử lý kịp
            "facingMode": "environment",
        },
        "audio": False,
    },
    async_processing=True, # Quan trọng để không làm đứng giao diện Streamlit
)

if ctx.state.playing:
    st.success("Đang kết nối thành công!")
