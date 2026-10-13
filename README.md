# 🪖 HelmetAI — Helmet Detection & Road Safety Analytics

A computer-vision web application that analyzes traffic images and videos to identify **riders with helmets** and **riders without helmets**. Built with Python, Ultralytics YOLO, OpenCV, and Streamlit.

<p align="center">
  <a href="https://helmet-detection-vileka-ai.streamlit.app/"><strong>🚀 Try the Live Demo</strong></a>
  &nbsp; • &nbsp;
  <a href="https://github.com/Vileka06/Helmet-Detection"><strong>💻 View Source Code</strong></a>
</p>

> **Project status:** Deployed Streamlit application. Detection quality depends on the trained model, image quality, lighting, camera angle, and object size. Live-camera behavior may vary by browser and network.

## ✨ Features

- **Image detection:** Upload a JPG, JPEG, PNG, or WEBP image and view the annotated result.
- **Video detection:** Upload a supported video and export an annotated MP4.
- **Live camera:** Use your browser camera for live detection where WebRTC connectivity is available.
- **Detection counts:** View counts for “With Helmet” and “Without Helmet” detections.
- **Adjustable inference:** Change the confidence threshold and image size from the sidebar.
- **Download results:** Save annotated images and processed videos.
- **Detection visualization:** Display bounding boxes and confidence scores on detected objects.

## 🖥️ Live Demo

**[Open HelmetAI](https://helmet-detection-vileka-ai.streamlit.app/)**

1. Open the app.
2. Choose **Upload Image**, **Upload Video**, or **Live Camera**.
3. Upload a supported file or grant camera permission.
4. Review the annotated output and detection counts.
5. Download the result when the download option is available.

For the most reliable first test, start with a clear traffic image containing visible motorcycle riders.

## 🧠 Model

The application loads its trained Ultralytics model from:

```text
models/best.pt
```

The app is configured to recognize these labels:

| Class ID | Label |
| --- | --- |
| 0 | With Helmet |
| 1 | Without Helmet |

The confidence threshold and inference image size can affect the number of detections. A low threshold can increase detections but may also increase false positives.

**Important:** This repository does not claim a particular accuracy, precision, recall, or mAP score. Add measured evaluation results only after testing the model on a separate labelled test set.

## 🧰 Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Application logic |
| Ultralytics YOLO | Object-detection inference |
| OpenCV | Image/video processing and bounding-box drawing |
| NumPy | Frame and image array handling |
| Pillow | Image loading |
| Streamlit | Interactive web interface |
| Streamlit WebRTC + PyAV | Browser live-camera streaming |
| Streamlit Community Cloud | Web deployment |

## 📁 Project Structure

```text
Helmet-Detection/
├── app.py                 # Streamlit interface and detection pipeline
├── predict.py             # Prediction utility
├── test_model.py          # Model testing utility
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── models/
    ├── best.pt            # Trained model weights
    └── README.md          # Model file notes
```

Some files or folders may differ if the repository is updated.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vileka06/Helmet-Detection.git
cd Helmet-Detection
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
. .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Check the model file

Make sure the trained weights exist at `models/best.pt`. The filename and folder path must match exactly.

### 5. Start the app

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, that you can open in your browser.

## ☁️ Deployment

This project is configured for **Streamlit Community Cloud**.

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app and connect the `Vileka06/Helmet-Detection` GitHub repository.
3. Select the `main` branch and `app.py` as the app entry point.
4. Deploy and check the build logs if startup fails.

The live-camera tab needs browser camera permission and a working WebRTC connection. A visible camera preview does not by itself guarantee that frames are reaching the server-side model for inference.

## 🧪 Testing and Evaluation

To evaluate a model fairly, test it on images that were not used for training. Useful object-detection metrics include:

- **Precision:** How many predicted detections are correct.
- **Recall:** How many labelled objects the model finds.
- **mAP:** A standard object-detection evaluation metric.
- **Confusion analysis:** Review false positives and missed detections for each class.

Keep the test dataset separate from the training data and report the actual results produced by your evaluation pipeline.

## ⚠️ Limitations

- Small, distant, blurred, or partly hidden riders can be difficult to detect.
- Lighting, camera angle, image resolution, and crowded scenes may affect predictions.
- Low confidence thresholds can produce false positives.
- Video processing can take time, particularly for long or high-resolution videos.
- Live-camera inference depends on browser permissions and WebRTC/network connectivity.
- This is an educational computer-vision project and should not be treated as the sole basis for road-safety enforcement decisions.

## 🔭 Possible Improvements

- Add a sample gallery with verified before-and-after detection results.
- Publish a reproducible evaluation report with precision, recall, and mAP.
- Improve detection on small and distant riders through dataset review and training.
- Add automated tests for model loading and image inference.
- Add clearer live-camera connection diagnostics and deployment checks.

## 👩‍💻 Author

**Vileka V** — Computer Science and Engineering student interested in Python, machine learning, and computer vision.

- GitHub: [@Vileka06](https://github.com/Vileka06)
- Project repository: [Helmet-Detection](https://github.com/Vileka06/Helmet-Detection)
- Live app: [HelmetAI on Streamlit](https://helmet-detection-vileka-ai.streamlit.app/)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
