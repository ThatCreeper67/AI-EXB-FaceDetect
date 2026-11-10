"""
Face Database Manager
Stores face embeddings with associated names for recognition.
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

DB_DIR = Path(__file__).parent / "database"
DB_FILE = DB_DIR / "faces.json"
EMBEDDINGS_DIR = DB_DIR / "embeddings"

# Ensure directories exist
DB_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)


class FaceDatabase:
    """Manages storage and retrieval of face embeddings with names."""

    def __init__(self):
        self.faces: Dict[str, List[np.ndarray]] = {}
        self.load_database()

    def load_database(self) -> None:
        """Load face database from disk."""
        if DB_FILE.exists():
            try:
                with open(DB_FILE, 'r') as f:
                    data = json.load(f)
                    for name, embeddings in data.items():
                        self.faces[name] = [np.array(emb, dtype=np.float32) for emb in embeddings]
            except Exception as e:
                print(f"Warning: Could not load face database: {e}")
                self.faces = {}

    def save_database(self) -> None:
        """Save face database to disk."""
        try:
            data = {}
            for name, embeddings in self.faces.items():
                data[name] = [emb.tolist() for emb in embeddings]

            with open(DB_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save face database: {e}")

    def add_face(self, name: str, embedding: np.ndarray) -> None:
        """Add a face embedding with a name."""
        if name not in self.faces:
            self.faces[name] = []
        self.faces[name].append(embedding)
        self.save_database()

    def get_all_faces(self) -> List[Tuple[str, np.ndarray]]:
        """Get all face embeddings with their names.

        Returns:
            List of tuples: (name, embedding)
        """
        all_faces = []
        for name, embeddings in self.faces.items():
            for emb in embeddings:
                all_faces.append((name, emb))
        return all_faces

    def recognize_face(self, embedding: np.ndarray, threshold: float = 0.9) -> Tuple[Optional[str], Optional[float]]:
        """Recognize a face by comparing with stored embeddings.

        Uses Euclidean distance for better separation between different people.

        Args:
            embedding: The face embedding to recognize
            threshold: Minimum cosine similarity score to consider a match

        Returns:
            Tuple of (name, confidence_score) or (None, None) if not recognized
        """
        if not self.faces:
            return None, None

        # Calculate Euclidean distance to all stored faces
        best_match = None
        best_distance = float('inf')
        best_cosine = -1

        for name, stored_embeddings in self.faces.items():
            for stored_emb in stored_embeddings:
                dist = self._euclidean_distance(embedding, stored_emb)
                cos_sim = self._cosine_similarity(embedding, stored_emb)

                if dist < best_distance:
                    best_distance = dist
                    best_match = name
                    best_cosine = cos_sim

        # Recognize only if cosine similarity is high enough
        if best_match and best_cosine > threshold:
            return (best_match, best_cosine)
        else:
            return (None, None)

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings."""
        return float(np.linalg.norm(a - b))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def get_registered_names(self) -> List[str]:
        """Get list of all registered names."""
        return list(self.faces.keys())

    def is_name_registered(self, name: str) -> bool:
        """Check if a name is already registered."""
        return name in self.faces and len(self.faces[name]) > 0


# Global database instance
face_db = FaceDatabase()
