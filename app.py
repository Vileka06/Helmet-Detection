import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="HelmetAI | Road Safety Analytics", page_icon="🪖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --slate:#384959; --navy:#253746; --light:#BDDDFC; --sky:#88BDF2; --text:#293744; --muted:#596B7A; --border:#D6E0E8; --soft:#F4F7FA; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background:#FFFFFF; color:var(--text); }
[data-testid="stHeader"] { background:#FFFFFF; }
.block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { color:#293744 !important; font-weight:800 !important; }
p,label { color:#293744; }
[data-testid="stCaptionContainer"] p { color:#596B7A !important; }
[data-testid="stSidebar"] { background:var(--slate); }
[data-testid="stSidebar"] * { color:#FFFFFF; }
[data-testid="stSidebar"] hr { border-color:#657787; }
.stButton > button, .stDownloadButton > button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
button[kind="primary"], button[kind="secondary"] {
  background-color:#253746 !important; color:#FFFFFF !important;
  border:2px solid #253746 !important; border-radius:10px !important;
  font-weight:800 !important; min-height:44px !important;
}
.stButton > button *, .stDownloadButton > button * { color:#FFFFFF !important; }
.stButton > button:hover, .stDownloadButton > button:hover,
button[kind="primary"]:hover, button[kind="secondary"]:hover {
  background:#52718A !important; border-color:#52718A !important; color:#FFFFFF !important;
}
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#F1F5F9; padding:8px; border-radius:12px; }
.stTabs [data-baseweb="tab"] { background:#E5EDF5; color:#253746 !important; border-radius:9px; padding:12px 20px; font-weight:800 !important; }
.stTabs [aria-selected="true"] { background:#BDDDFC !important; color:#253746 !important; }
[data-testid="stMetric"] { background:#F4F7FA; border:1px solid #D6E0E8; padding:18px; border-radius:12px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#253746 !important; font-weight:800 !important; }
.helmet-title { font-size:34px; font-weight:800; color:#253746; margin-bottom:5px; }
.helmet-subtitle { color:#596B7A; font-size:15px; margin-bottom:24px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:26px;font-weight:800;color:white'>🪖 HelmetAI</div><div style='font-size:11px;letter-spacing:1.5px;color:#E3EDF5'>ROAD SAFETY ANALYTICS</div>", unsafe_allow_html=True)
    st.divider()
    st.subheader("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 480, 640, 800, 960], value=640)
    st.caption("Higher image sizes may improve small-object detection but take longer.")

st.markdown("<div class='helmet-title'>Helmet Detection & Traffic Safety</div><div class='helmet-subtitle'>Analyze traffic images and videos or use your live camera to identify helmet detections.</div>", unsafe_allow_html=True)

MODEL_PATH = Path("models/best.pt")

@st.cache_resource
def load_model(path):
    return YOLO(path)

if not MODEL_PATH.exists():
    st.error("Trained model not found: models/best.pt")
    st.markdown("Upload your trained best.pt model to the repository under models/best.pt, then redeploy. Detection cannot run without this model file.")
    st.stop()

model = load_model(str(MODEL_PATH))

@st.cache_resource
def load_person_model():
    return YOLO("yolov8n.pt")

def person_boxes(frame):
    try:
        result = load_person_model().predict(frame, conf=0.20, imgsz=640, classes=[0], verbose=False)[0]
        return [tuple(map(int, box.xyxy[0].cpu().numpy().tolist())) for box in result.boxes]
    except Exception:
        return []

def head_is_near_person(candidate, people):
    x1, y1, x2, y2 = candidate[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    for px1, py1, px2, py2 in people:
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            continue
        if px1 - .05 * pw <= cx <= px2 + .05 * pw and py1 <= cy <= py1 + .43 * ph:
            if cw <= .90 * pw and ch <= .50 * ph:
                return True
    return False

def draw_box(img, x1, y1, x2, y2, label, score):
    h, w = img.shape[:2]
    color = (32, 155, 65) if label.lower() == "with helmet" else (0, 45, 220)
    thick = max(2, int(min(w, h) / 450))
    font = max(.65, min(1.15, w / 1100))
    text = f"{label} {score:.2f}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
    top = max(0, y1 - th - base - 8)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv2.rectangle(img, (x1, top), (min(w - 1, x1 + tw + 14), top + th + base + 8), color, -1)
    cv2.putText(img, text, (x1 + 7, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font, (255,255,255), thick, cv2.LINE_AA)

def run_detection(frame, tiled=False):
    height, width = frame.shape[:2]
    people = person_boxes(frame)
    candidates = []
    if tiled and width > 1 and height > 1:
        tw, th = max(1, int(width * .65)), max(1, int(height * .65))
        xs, ys = sorted(set([0, max(0, width - tw)])), sorted(set([0, max(0, height - th)]))
        for y in ys:
            for x in xs:
                crop = frame[y:min(y+th,height), x:min(x+tw,width)]
                if crop.size == 0:
                    continue
                result = model.predict(crop, conf=confidence, imgsz=max(640, image_size), verbose=False)[0]
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                    coords = [max(0,int(bx1+x)), max(0,int(by1+y)), min(width-1,int(bx2+x)), min(height-1,int(by2+y))]
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    label = str(result.names.get(cls_id, cls_id)).replace("_"," ")
                    if label.lower() in ("with helmet","without helmet") and (not people or head_is_near_person(coords, people)):
                        candidates.append((*coords, score, cls_id, label))
    else:
        result = model.predict(frame, conf=confidence, imgsz=image_size, verbose=False)[0]
        for box in result.boxes:
            coords = list(map(int, box.xyxy[0].cpu().numpy().tolist()))
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = str(result.names.get(cls_id, cls_id)).replace("_"," ")
            if label.lower() in ("with helmet","without helmet") and (not people or head_is_near_person(coords, people)):
                candidates.append((*coords, score, cls_id, label))
    kept = []
    for cls_id in sorted(set(c[5] for c in candidates)):
        group = [c for c in candidates if c[5] == cls_id]
        boxes_xywh = [[c[0],c[1],max(1,c[2]-c[0]),max(1,c[3]-c[1])] for c in group]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, [float(c[4]) for c in group], float(confidence), .45)
        if len(idxs):
            kept.extend(group[int(i)] for i in np.array(idxs).reshape(-1))
    annotated = frame.copy()
    counts = {"With Helmet":0, "Without Helmet":0}
    for x1,y1,x2,y2,score,cls_id,label in kept:
        key = "With Helmet" if label.lower() == "with helmet" else "Without Helmet"
        counts[key] += 1
        draw_box(annotated,x1,y1,x2,y2,key,score)
    return annotated, counts

def show_metrics(counts):
    left, right = st.columns(2)
    left.metric("With helmet", counts.get("With Helmet",0))
    right.metric("Without helmet", counts.get("Without Helmet",0))

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Upload Image", "🎞️ Upload Video", "📷 Live Camera"])

with tab_img:
    st.subheader("Image analysis")
    file = st.file_uploader("Choose a traffic image", type=["jpg","jpeg","png","webp"], key="img")
    if file and st.button("Detect helmets in image", type="primary", key="detect_img"):
        pil = Image.open(file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        with st.spinner("Analyzing image..."):
            annotated, counts = run_detection(frame)
        left, right = st.columns(2)
        left.image(pil, caption="Original image", use_container_width=True)
        right.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB), caption="Detection result", use_container_width=True)
        show_metrics(counts)
        ok, enc = cv2.imencode(".jpg", annotated)
        if ok:
            st.download_button("Download annotated image", enc.tobytes(), "helmet_detection_result.jpg", "image/jpeg")

with tab_vid:
    st.subheader("Video processing")
    st.write("Upload a traffic video to create an annotated MP4.")
    video = st.file_uploader("Choose a video", type=["mp4","mov","avi","m4v"], key="vid")
    max_seconds = st.slider("Maximum video duration (seconds)", 10, 180, 60, 10)
    frame_step = st.select_slider("Analyze every Nth frame", options=[1,2,3,5], value=2)
    if video and st.button("Process video", type="primary", key="process_vid"):
        input_path = output_path = None
        cap = writer = None
        try:
            suffix = Path(video.name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(video.getvalue())
                input_path = tmp.name
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open video. Try an MP4 file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not 1 <= fps <= 120:
                    fps = 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                max_frames = min(total if total > 0 else int(fps*max_seconds), int(fps*max_seconds))
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    output_path = tmp.name
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
                progress = st.progress(0)
                status = st.empty()
                totals = {"With Helmet":0,"Without Helmet":0}
                analyzed = idx = 0
                while idx < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % frame_step == 0:
                        annotated, counts = run_detection(frame, tiled=True)
                        totals["With Helmet"] += counts["With Helmet"]
                        totals["Without Helmet"] += counts["Without Helmet"]
                        analyzed += 1
                    else:
                        annotated = frame
                    writer.write(annotated)
                    idx += 1
                    if idx % 5 == 0 or idx == max_frames:
                        progress.progress(min(1.0, idx / max(1,max_frames)))
                        status.write(f"Processed {idx} / {max_frames} frames")
                cap.release()
                writer.release()
                cap = writer = None
                if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    st.success("Video processing completed.")
                    st.video(output_path)
                    a,b,c = st.columns(3)
                    a.metric("Frames written", idx)
                    b.metric("Frames analyzed", analyzed)
                    c.metric("Without-helmet detections*", totals["Without Helmet"])
                    st.caption("*Counts detections across analyzed frames; the same rider may be counted more than once.")
                    with open(output_path,"rb") as f:
                        st.download_button("Download processed MP4", f.read(), "helmet_detection_video.mp4", "video/mp4")
                else:
                    st.error("The processed video could not be created.")
        except Exception as e:
            st.error(f"Video processing failed: {e}")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

with tab_cam:
    st.subheader("Live helmet detection")
    st.write("Allow camera access in your browser to analyze the live feed.")
    st.info("Live camera requires browser camera permission and an HTTPS deployment or localhost.")
    try:
        import av
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        class HelmetCameraProcessor(VideoProcessorBase):
            def __init__(self):
                self.threshold = confidence
                self.size = image_size
            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                annotated = image.copy()
                found = 0
                try:
                    # Run the trained helmet detector directly on each camera frame.
                    # Avoid the extra person-detector filter here because close-up webcam
                    # views often contain only a head/shoulders and fail that filter.
                    # Reject very weak predictions in live camera frames. A low threshold
                    # caused many false positives on faces and clothing.
                    camera_confidence = 0.45
                    result = model.predict(
                        image, conf=camera_confidence, iou=0.45,
                        imgsz=max(640, self.size), max_det=20, verbose=False
                    )[0]
                    raw_count = len(result.boxes)
                    best_score = 0.0
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())
                        best_score = max(best_score, score)
                        label = {0: "With Helmet", 1: "Without Helmet"}.get(
                            cls_id, str(result.names.get(cls_id, cls_id)).replace("_", " ")
                        )
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
                        draw_box(annotated, x1, y1, x2, y2, label, score)
                        found += 1
                    # Diagnostic overlay lets us distinguish model output from a camera issue.
                    status = f"Detections: {raw_count} | max confidence: {best_score:.2f} | threshold: {camera_confidence:.2f}"
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (35, 55, 72), -1)
                    cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    if found == 0:
                        cv2.putText(annotated, "No confident helmet detections in this frame",
                            (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as exc:
                    message = f"Detection error: {str(exc)[:90]}"
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 48), (0, 0, 180), -1)
                    cv2.putText(annotated, message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        webrtc_streamer(
            key="helmetai-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HelmetCameraProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            async_processing=True,
        )
    except ImportError:
        st.error("Live camera packages are missing. Install streamlit-webrtc and av.")

st.divider()
st.caption("Detection quality depends on model training, scene scale, lighting, and camera angle.")
