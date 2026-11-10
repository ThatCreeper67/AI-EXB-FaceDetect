"""
Real-time Face Detection with OpenCV DNN (SSD Caffe Model)
Detects faces from webcam and draws bounding boxes (hitboxes) around them.
Press 'q' to quit.
"""
import cv2
import numpy as np
import os
import sys

from cuda_render import (
    close_cuda_renderer,
    cuda_overlay_available,
    cuda_overlay_status,
    render_overlays_cuda,
)
from face_database.face_db import face_db

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "face_recognizer.onnx")

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Face recognition: input size and output embedding dimension for the ONNX model.
RECOGNIZER_SIZE = 112
EMBEDDING_DIM = 128


class FaceEmbedder:
    """Convert a cropped face image into a 128-dim embedding for recognition.

    The ONNX face_recognizer model expects a 112x112 BGR image normalized to
    [-1, 1] (mean 127.5, scale 1/127.5, channels swapped to RGB).
    """

    def __init__(self, model_path=RECOGNIZER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face recognizer model not found: {model_path}")
        self.net = cv2.dnn.readNet(model_path)
        print(f"Loaded face recognizer: {model_path}", flush=True)

    def get_embedding(self, face_bgr):
        """Return a normalized 128-dim embedding list for a BGR face crop."""
        resized = cv2.resize(face_bgr, (RECOGNIZER_SIZE, RECOGNIZER_SIZE))
        blob = cv2.dnn.blobFromImage(
            resized, 1.0 / 127.5, (RECOGNIZER_SIZE, RECOGNIZER_SIZE),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        self.net.setInput(blob)
        embedding = self.net.forward()
        return embedding.flatten()

    def recognize_face(self, face_bgr, threshold=0.9) -> Tuple[Optional[str], Optional[float]]:
        """Recognize a face and return the name and confidence if recognized.

        Args:
            face_bgr: BGR face image
            threshold: Minimum cosine similarity score to consider a match

        Returns:
            Tuple of (name, confidence) or (None, None) if not recognized
        """
        embedding = self.get_embedding(face_bgr)
        name, confidence = face_db.recognize_face(embedding, threshold)
        return name, confidence

def load_model():
    """Load the Caffe face detection model."""
    if not os.path.exists(PROTO_PATH):
        sys.exit(f"Error: {PROTO_PATH} not found")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: {MODEL_PATH} not found")

    # OpenCV 5.x uses readNet for all formats
    net = cv2.dnn.readNet(MODEL_PATH, PROTO_PATH)
    # Try to use CUDA if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA acceleration")
    else:
        print("Using CPU (CUDA not available)")
    print(cuda_overlay_status())
    return net

def cosine_similarity(a, b):
    """Cosine similarity between two embedding vectors (0..1, 1 = identical)."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def detect_faces(net, frame):
    """Run face detection on a frame, return list of (x1, y1, x2, y2, confidence)."""
    h, w = frame.shape[:2]

    # Create blob from image (resize to 300x300, normalize)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > CONFIDENCE_THRESHOLD:
            # Scale bounding box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = (int(value) for value in box.astype("int"))

            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            faces.append((x1, y1, x2, y2, confidence))

    return faces

def draw_hitboxes(frame, faces, face_count=None, names=None, confidences=None):
    """Draw bounding boxes and labels on the frame.

    Args:
        frame: Input frame
        faces: List of face bounding boxes [(x1,y1,x2,y2,conf), ...]
        face_count: Optional count of faces
        names: Optional list of names for each face
        confidences: Optional list of confidence scores for each face
    """
    if names is None:
        names = [None] * len(faces)
    if confidences is None:
        confidences = [None] * len(faces)

    if cuda_overlay_available():
        # CUDA path draws confidence labels only; draw names afterwards in CPU.
        frame = render_overlays_cuda(frame, faces, face_count)
        for (x1, y1, x2, y2, _conf), name, conf in zip(faces, names, confidences):
            if name:
                _draw_name_label(frame, x1, y1, x2, name, conf=conf)
            else:
                _draw_name_label(frame, x1, y1, x2, None, conf=conf)
        return frame

    for (x1, y1, x2, y2, conf), name, face_conf in zip(faces, names, confidences):
        # Draw the bounding box (hitbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw corner markers for "hitbox" effect
        corner_len = 20
        color = (0, 255, 0)
        thickness = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # Draw label: registered name with confidence or unregistered face confidence
        if name:
            _draw_name_label(frame, x1, y1, x2, name, conf=face_conf)
        else:
            _draw_name_label(frame, x1, y1, x2, None, conf=face_conf)

    # Draw face count only once at top-left (avoid overlap with CUDA path which also draws it)
    if face_count is not None:
        cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def _draw_name_label(frame, x1, y1, x2, name, conf=None):
    """Draw the registered name (or confidence) above a face box.

    Args:
        frame: Input frame
        x1, y1, x2, y2: Face bounding box coordinates
        name: Registered name (e.g., "AARUSH") or "TestUser" for unregistered
        conf: Confidence score for recognition
    """
    if name:
        # Show registered name with confidence if available
        if conf is not None:
            label = f"{name} ({conf:.2f})"
        else:
            label = name
        text_color = (0, 0, 0)
    elif conf is not None:
        label = f"Face: {conf:.2f}"
        text_color = (0, 0, 0)
    else:
        return

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                  (x1 + label_size[0], y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def main():
    print("Loading face detection model...")
    net = load_model()

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit("Error: Could not open webcam. Try changing the camera index (0, 1, 2...).")

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting face detection. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            # Mirror for selfie-view
            frame = cv2.flip(frame, 1)

            # Detect faces
            faces = detect_faces(net, frame)

            # Draw hitboxes
            frame = draw_hitboxes(frame, faces, face_count=len(faces))

            # Display
            cv2.imshow("Face Detection - Hitboxes", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_cuda_renderer()
    print("Done.")

if __name__ == "__main__":
    main()