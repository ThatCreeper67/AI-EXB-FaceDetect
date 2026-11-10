# DataCam — Real-time Face Detection Web Stream

A school exhibition project that streams live webcam face detection from a host computer to any browser on the same Wi-Fi network. Built with **OpenCV DNN (SSD Caffe)**, **Flask**, and **WebSockets**.

---

## What It Does

- Captures webcam video on the host machine
- Runs real-time face detection using OpenCV's SSD model (Caffe)
- Streams annotated frames (with bounding boxes + confidence scores) to connected browsers over WebSockets
- Serves a clean web UI at `http://<host-ip>:5000` showing the live feed
- Includes a **"Capture & analyze"** button that triggers a fresh model inference on the latest frame and prints results to the server terminal

---

## Quick Start

### Prerequisites

- Python 3.9+
- A webcam (index 0)
- Windows, macOS, or Linux

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` doesn't exist yet, install manually:
> ```bash
> pip install opencv-python flask websockets numpy
> ```

### Run the server

```bash
cd "C:\Nerd\AI EXB\DataCam"
python server.py
```

You'll see output like:

```
Using camera backend: MSMF
  Waiting for WebSocket server to start...
  WebSocket server ready on port 8765

  Face-detection stream is starting up.
  Flask listening on 0.0.0.0:5000
  Open this exact URL on your phone: http://192.168.1.42:5000/
  WebSocket endpoint:                ws://192.168.1.42:8765
  Keep the phone on the same Wi-Fi network; allow Python through the firewall if prompted.

  Press 'q' in the OpenCV window to stop the camera feed.
```

### View the stream

1. **On the host machine:** An OpenCV window titled "Face Detection - Server" opens automatically showing the local preview.
2. **On any other device (phone, laptop):** Open the printed LAN URL (e.g., `http://192.168.1.42:5000/`) in a browser. Both devices must be on the same Wi-Fi network.

---

## Project Structure

```
DataCam/
├── server.py              # Main entry point: Flask + WebSocket + camera thread
├── face_detect.py         # OpenCV DNN face detection (SSD Caffe model)
├── requirements.txt       # Python dependencies (create if missing)
├── models/
│   ├── deploy.prototxt    # Caffe model architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained weights
└── templates/
    └── index.html         # Web UI (canvas + WebSocket client)
```

---

## How It Works

| Component | Role |
|-----------|------|
| **Camera thread** | Grabs frames from webcam, runs face detection, draws hitboxes, pushes JPEG frames + face data to a thread-safe queue |
| **WebSocket server** (asyncio, port 8765) | Broadcasts frames (binary) + face metadata (JSON) to all connected browsers; handles `"capture"` messages for on-demand inference |
| **Flask** (port 5000) | Serves the static HTML page |
| **Browser client** | Renders frames to `<canvas>`, draws bounding boxes from face metadata, provides "Capture & analyze" button |

### Data flow

```
Webcam → Camera thread (detect + draw) → Queue → WebSocket pump → Browser canvas
                                          ↓
                                   Flask serves index.html
```

---

## Model Details

- **Architecture:** SSD (Single Shot Detector) with ResNet-10 backbone
- **Input size:** 300×300
- **Framework:** Caffe (loaded via `cv2.dnn.readNet`)
- **Confidence threshold:** 0.5 (adjustable in `face_detect.py:CONFIDENCE_THRESHOLD`)
- **GPU acceleration:** Automatically uses CUDA if `cv2.cuda` is available

The model files (`deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`) are included in `models/`.

---

## Configuration

Edit these constants in `server.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | 5000 | HTTP port for the web UI |
| `WS_PORT` | 8765 | WebSocket port for frame streaming |
| `FRAME_W`, `FRAME_H` | 640, 480 | Capture resolution |

In `face_detect.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence to keep a detection |

---

## Common Issues

| Problem | Fix |
|---------|-----|
| "Could not open webcam" | Try a different camera index in `face_detect.py` (`cv2.VideoCapture(1)` etc.) or check permissions |
| Phone can't connect | Ensure both devices are on the same Wi-Fi; allow Python through Windows Firewall (prompt appears on first run); verify the LAN IP printed at startup |
| WebSocket connection fails | Check port 8765 isn't blocked; try `ws://<host-ip>:8765` directly in browser console |
| No face detections | Lower `CONFIDENCE_THRESHOLD`; ensure good lighting; model works best on frontal faces |
| CUDA not detected | Install `opencv-python-headless` + CUDA toolkit; or accept CPU mode (printed at startup) |

---

## Stopping the Server

- Press **`q`** in the OpenCV window, **or**
- Press **`Ctrl+C`** in the terminal

Both will cleanly release the camera and close windows.

---

## License

School exhibition project — educational use.
