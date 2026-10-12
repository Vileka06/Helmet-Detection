# HelmetAI — Helmet Detection & Road Safety

Streamlit app for helmet detection in traffic images, videos, and a live camera stream.

## Features
- Upload an image, inspect annotated detections, and download the result.
- Upload a video, process sampled frames using tiled inference, and download annotated MP4.
- Live browser camera detection using Streamlit WebRTC.
- High-contrast labels, buttons, tabs, and metric cards.

## Trained model
The app loads the trained YOLO weights from `models/best.pt`. Keep that exact path and filename when updating the model.

## Run locally
Install requirements with `python -m pip install -r requirements.txt`, then run `streamlit run app.py`.

## Deploy
Deploy the `main` branch on Streamlit Community Cloud with `app.py` as the main file. The live-camera feature requires browser camera permissions and an HTTPS deployment.

Detection quality depends on model training, scene scale, lighting, and camera position.
