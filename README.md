# HelmetAI — Helmet Detection & Road Safety

Streamlit app for helmet detection in traffic images, videos, and a live camera stream.

## Features
- Upload an image, inspect annotated detections, and download the result.
- Upload a video, process sampled frames using tiled inference, and download annotated MP4.
- Live browser camera detection using Streamlit WebRTC.
- High-contrast labels, buttons, tabs, and metric cards.

## Required trained model
The application expects your trained YOLO model at models/best.pt. This repository was recreated after deletion and the binary weights were not restored by the GitHub text-file upload. Upload your own trained best.pt file to the repository at models/best.pt before using the app. Do not replace it with an unrelated generic model.

## Run locally
Install requirements with `python -m pip install -r requirements.txt`, then run `streamlit run app.py`.

## Deploy
Deploy this repository's main branch on Streamlit Community Cloud with app.py as the main file. Add models/best.pt first. The live-camera tab requires browser camera permissions and an HTTPS deployment.

Detection quality depends on model training, scene scale, lighting, and camera position.
