"""Stream webcam face detection to browser clients over Flask and WebSockets.

The webcam is attached to this computer. Open the printed LAN URL on a
phone connected to the same Wi-Fi network to view the mirrored live feed.
The browser's Capture button reruns detection and prints its results here.
"""
import asyncio
import io
import json
import queue
import socket
import sys
import threading
import time
import base64
import os

import cv2
import numpy as np
import websockets
from flask import Flask, render_template, request, jsonify, send_file

from cuda_render import close_cuda_renderer
from face_detect import load_model, detect_faces, draw_hitboxes, FaceEmbedder
from face_database.face_db import face_db

FLASK_PORT = 5000
WS_PORT = 8765
FRAME_W, FRAME_H = 640, 480

app = Flask(__name__)

# Producer/consumer queue: camera thread -> asyncio broadcast loop.
# Each item is {"frame": jpeg bytes, "faces": [(x1,y1,x2,y2,conf), ...]}
frame_queue = queue.Queue()

# Module-level camera handle, set in main().
camera = None
connected = set()

# Face embedder for recognition
face_embedder = None

# Synchronization event to signal WebSocket server is ready
ws_server_ready = threading.Event()


class Camera:
    """Continuous webcam capture + face detection in a background thread."""

    def __init__(self):
        self.net = load_model()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(0, backend)
        if not self.cap.isOpened():
            backend_name = "DirectShow" if sys.platform == "win32" else "default"
            raise RuntimeError(f"Could not open webcam (index 0, {backend_name} backend)")
        print(f"Using camera backend: {self.cap.getBackendName()}", flush=True)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.latest_frame = None  # raw BGR numpy frame (for capture model runs)
        self.faces = []
        self.running = True
        self.frame_count = 0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Mirror for selfie-view, matching the original script.
        return cv2.flip(frame, 1)

    def run(self):
        """Grab frames forever, detect faces, queue them for broadcast, and show window."""
        cv2.namedWindow("Face Detection - Server", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Detection - Server", FRAME_W, FRAME_H)

        try:
            while self.running:
                frame = self.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                with self.lock:
                    self.latest_frame = frame.copy()

                with self.inference_lock:
                    faces = detect_faces(self.net, frame)

                # Recognize faces for better accuracy (every frame, but with performance consideration)
                names = [None] * len(faces)
                confidences = [None] * len(faces)

                if face_embedder:
                    for i, (x1, y1, x2, y2, conf) in enumerate(faces):
                        face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                        max(0, x1):min(frame.shape[1], x2)]
                        if face_crop.size > 0:
                            recognized_name, recognition_conf = face_embedder.recognize_face(face_crop, threshold=0.9)
                            names[i] = recognized_name if recognized_name else "TestUser"
                            confidences[i] = recognition_conf if recognition_conf else 0.0

                self.frame_count += 1

                display_frame = draw_hitboxes(
                    frame.copy(), faces, face_count=len(faces), names=names, confidences=confidences
                )

                ok, jpeg = cv2.imencode(
                    ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not ok:
                    continue
                jpeg_bytes = jpeg.tobytes()

                with self.lock:
                    self.faces = faces
                    self.names = names
                    self.confidences = confidences

                frame_queue.put({"frame": jpeg_bytes, "faces": faces, "names": names})

                cv2.imshow("Face Detection - Server", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
        finally:
            close_cuda_renderer()
            cv2.destroyWindow("Face Detection - Server")

    def stop(self):
        self.running = False
        self.cap.release()


# ---------------------------------------------------------------------------
# WebSocket server (runs in its own asyncio loop on a background thread)
# ---------------------------------------------------------------------------
async def ws_handler(websocket):
    connected.add(websocket)
    try:
        async for message in websocket:
            if message == "capture":
                await handle_capture(websocket)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected.discard(websocket)


async def handle_capture(websocket):
    """Trigger a model run on the latest frame; print results to terminal."""
    active_camera = camera
    if active_camera is None:
        await websocket.send(json.dumps({
            "type": "capture",
            "error": "camera is not ready",
        }))
        return

    with active_camera.lock:
        frame = active_camera.latest_frame
    if frame is None:
        await websocket.send(json.dumps({"type": "capture", "error": "no frame yet"}))
        return

    # Fresh model run triggered by the capture button.
    with active_camera.inference_lock:
        fresh_faces = detect_faces(active_camera.net, frame)
    confidences = [round(float(f[4]), 3) for f in fresh_faces]

    print(
        f"[Capture] model run complete — faces: {len(fresh_faces)} "
        f"| confidences: {confidences}",
        flush=True,
    )

    await websocket.send(json.dumps({
        "type": "capture",
        "faces": len(fresh_faces),
        "confidences": confidences,
    }))


@app.route("/api/register_face", methods=["POST"])
def register_face():
    """Register a new face with a name."""
    try:
        data = request.json
        if not data or 'name' not in data or 'face_data' not in data:
            return jsonify({"success": False, "error": "Missing name or face_data"}), 400

        name = data['name'].strip()
        face_data = data['face_data']

        if not name:
            return jsonify({"success": False, "error": "Name cannot be empty"}), 400

        # Decode face image from base64
        face_bytes = base64.b64decode(face_data.split(',')[1])
        nparr = np.frombuffer(face_bytes, np.uint8)
        face_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_bgr is None:
            return jsonify({"success": False, "error": "Invalid face image"}), 400

        # Get embedding and save to database
        embedding = face_embedder.get_embedding(face_bgr)
        face_db.add_face(name, embedding)

        return jsonify({
            "success": True,
            "message": f"Face registered as {name}",
            "name": name
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/get_registered_faces", methods=["GET"])
def get_registered_faces():
    """Get list of all registered faces."""
    try:
        names = face_db.get_registered_names()
        return jsonify({
            "success": True,
            "faces": names
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/get_face_image", methods=["GET"])
def get_face_image():
    """Serve a sample face image for a registered person.

    In a real system, you would serve actual face crops from the database.
    For now, we return a placeholder image.
    """
    name = request.args.get("name", "")

    if not name:
        return "Name parameter required", 400

    # Create a simple placeholder image with the name
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img.fill(30)  # Dark background

    # Add text
    cv2.putText(img, f"Sample: {name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Convert to JPEG and send
    _, buffer = cv2.imencode('.jpg', img)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg'
    )


async def pump():
    """Drain the frame queue and broadcast to all connected clients."""
    loop = asyncio.get_running_loop()
    while True:
        data = await loop.run_in_executor(None, frame_queue.get)
        if not connected:
            continue
        frame_msg = data["frame"]
        faces_msg = json.dumps({"type": "frame", "faces": data["faces"]})
        # Send to all connected clients
        await asyncio.gather(
            *[ws.send(frame_msg) for ws in connected],
            *[ws.send(faces_msg) for ws in connected],
            return_exceptions=True
        )


async def main_ws():
    """Run the WebSocket server."""
    server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    # Signal that WebSocket server is ready
    ws_server_ready.set()
    print(f"  WebSocket server listening on 0.0.0.0:{WS_PORT}", flush=True)
    # Keep the server running while pumping frames
    await pump()


def start_ws_server():
    """Start WebSocket server in a new thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_ws())


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def get_lan_ipv4():
    """Find the IPv4 address other devices can use to reach this computer."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            if not address.startswith("127."):
                return address
    except OSError:
        pass

    try:
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return None

    for ip in addresses:
        if not ip.startswith(("127.", "169.254.")):
            return ip
    return None


def main():
    global camera, face_embedder

    # Initialize face embedder
    face_embedder = FaceEmbedder()

    camera = Camera()

    camera_thread = threading.Thread(target=camera.run, daemon=True)
    camera_thread.start()

    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    # Wait for WebSocket server to be ready before starting Flask
    print("  Waiting for WebSocket server to start...")
    ws_server_ready.wait(timeout=10)
    if ws_server_ready.is_set():
        print("  WebSocket server ready on port", WS_PORT)
    else:
        print("  WARNING: WebSocket server did not signal ready within 10 seconds")

    lan_ip = get_lan_ipv4()
    print()
    print("  Face-detection stream is starting up.")
    print(f"  Flask listening on 0.0.0.0:{FLASK_PORT}")
    if lan_ip:
        print(f"  Open this exact URL on your phone: http://{lan_ip}:{FLASK_PORT}/")
        print(f"  WebSocket endpoint:                ws://{lan_ip}:{WS_PORT}")
    else:
        print("  Could not detect a LAN IPv4 address.")
        print(f"  Flask is listening on port {FLASK_PORT}; find this computer's Wi-Fi IPv4 address")
        print(f"  and open http://<computer-ip>:{FLASK_PORT}/ on your phone.")
    print("  Keep the phone on the same Wi-Fi network; allow Python through the firewall if prompted.")
    print()
    print("  Press 'q' in the OpenCV window to stop the camera feed.")
    print()

    try:
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()