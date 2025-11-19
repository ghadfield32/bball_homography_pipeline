# api/src/cv/siglip_reid.py
"""
SigLIP-based visual re-identification for basketball players.

Uses SigLIP embeddings to compute visual similarity between player crops
for re-identification across camera cuts and track breaks.

Features:
- Extract embeddings from player bounding box crops
- Compute cosine similarity between embeddings
- Track embedding history with temporal weighting
- Re-ID matching with configurable thresholds

Usage:
    reid = SigLIPReID()
    reid.load_model()

    for frame_idx, frame in enumerate(video):
        tracked_dets = tracker.update(...)
        reid.update(frame, tracked_dets, frame_idx)

    # After track break, find match
    match = reid.find_best_match(new_track_id, lost_track_ids)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


@dataclass
class EmbeddingHistory:
    """Embedding history for a tracked player."""
    track_id: int

    # Embeddings over time (frame_idx, embedding)
    embeddings: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    max_history: int = 50

    # Aggregated embedding
    _mean_embedding: Optional[np.ndarray] = None
    _needs_update: bool = True

    def add_embedding(self, frame_idx: int, embedding: np.ndarray) -> None:
        """Add an embedding observation."""
        self.embeddings.append((frame_idx, embedding.copy()))

        # Trim history
        if len(self.embeddings) > self.max_history:
            self.embeddings.pop(0)

        self._needs_update = True

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Get mean embedding with recency weighting."""
        if not self.embeddings:
            return None

        if self._needs_update:
            # Compute weighted mean (more recent = higher weight)
            weights = np.array([0.5 + 0.5 * (i / len(self.embeddings))
                               for i in range(len(self.embeddings))])
            weights /= weights.sum()

            embeddings = np.array([e for _, e in self.embeddings])
            self._mean_embedding = np.average(embeddings, axis=0, weights=weights)

            # Normalize
            norm = np.linalg.norm(self._mean_embedding)
            if norm > 0:
                self._mean_embedding /= norm

            self._needs_update = False

        return self._mean_embedding

    @property
    def last_frame(self) -> int:
        """Last frame this track was observed."""
        return self.embeddings[-1][0] if self.embeddings else -1


@dataclass
class SigLIPReID:
    """
    SigLIP-based visual re-identification system.

    Uses vision-language model embeddings for appearance-based
    player re-identification across track breaks.
    """
    # Model configuration
    model_name: str = "google/siglip-base-patch16-224"
    similarity_threshold: float = 0.85
    embedding_dim: int = 768

    # Crop parameters
    crop_padding: float = 0.1
    crop_size: Tuple[int, int] = (224, 224)

    # Internal state
    _model = None
    _processor = None
    _track_embeddings: Dict[int, EmbeddingHistory] = field(default_factory=dict)
    _device: str = "cpu"

    def __post_init__(self):
        """Initialize state."""
        self._track_embeddings = {}

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load SigLIP model.

        Args:
            model_name: HuggingFace model name

        Returns:
            True if loaded successfully
        """
        if model_name:
            self.model_name = model_name

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            # Check for GPU
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model and processor
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self._device)
            self._model.eval()

            print(f"[INFO][siglip_reid] Loaded {self.model_name} on {self._device}")
            return True

        except ImportError as e:
            print(f"[WARN][siglip_reid] Missing dependencies: {e}")
            print("[WARN][siglip_reid] Install: pip install transformers torch")
            return False

        except Exception as e:
            print(f"[WARN][siglip_reid] Failed to load model: {e}")
            return False

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract padded crop from frame.

        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Resized crop or None
        """
        import cv2

        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1

        # Add padding
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        # Resize to model input size
        crop = cv2.resize(crop, self.crop_size)

        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        return crop

    def _compute_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute SigLIP embedding for a crop.

        Args:
            crop: RGB image crop

        Returns:
            Normalized embedding vector
        """
        if self._model is None or self._processor is None:
            return None

        try:
            import torch

            # Process image
            inputs = self._processor(images=crop, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            return embedding

        except Exception as e:
            print(f"[WARN][siglip_reid] Embedding failed: {e}")
            return None

    def update(
        self,
        frame: np.ndarray,
        tracked_detections,
        frame_idx: int,
    ) -> Dict[int, np.ndarray]:
        """
        Update embeddings for tracked detections.

        Args:
            frame: BGR image
            tracked_detections: sv.Detections with tracker_id
            frame_idx: Current frame number

        Returns:
            Dict mapping track_id -> embedding
        """
        results = {}

        if tracked_detections.tracker_id is None:
            return results

        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)
            bbox = tracked_detections.xyxy[i]

            # Extract crop
            crop = self._extract_crop(frame, bbox)
            if crop is None:
                continue

            # Compute embedding
            embedding = self._compute_embedding(crop)
            if embedding is None:
                continue

            # Update history
            if track_id not in self._track_embeddings:
                self._track_embeddings[track_id] = EmbeddingHistory(track_id=track_id)

            self._track_embeddings[track_id].add_embedding(frame_idx, embedding)
            results[track_id] = embedding

        return results

    def compute_similarity(
        self,
        track_id_a: int,
        track_id_b: int,
    ) -> float:
        """
        Compute similarity between two tracks.

        Args:
            track_id_a: First track ID
            track_id_b: Second track ID

        Returns:
            Cosine similarity (0-1)
        """
        if track_id_a not in self._track_embeddings or track_id_b not in self._track_embeddings:
            return 0.0

        emb_a = self._track_embeddings[track_id_a].get_mean_embedding()
        emb_b = self._track_embeddings[track_id_b].get_mean_embedding()

        if emb_a is None or emb_b is None:
            return 0.0

        # Cosine similarity (embeddings are normalized)
        return float(np.dot(emb_a, emb_b))

    def find_best_match(
        self,
        query_track_id: int,
        candidate_track_ids: List[int],
        min_similarity: Optional[float] = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Find best matching track from candidates.

        Args:
            query_track_id: Track to match
            candidate_track_ids: List of candidate tracks
            min_similarity: Minimum similarity threshold

        Returns:
            (best_track_id, similarity) or None if no match
        """
        if min_similarity is None:
            min_similarity = self.similarity_threshold

        if query_track_id not in self._track_embeddings:
            return None

        best_match = None
        best_sim = min_similarity

        for cand_id in candidate_track_ids:
            if cand_id == query_track_id:
                continue

            sim = self.compute_similarity(query_track_id, cand_id)
            if sim > best_sim:
                best_sim = sim
                best_match = cand_id

        if best_match is not None:
            return (best_match, best_sim)

        return None

    def get_lost_tracks(
        self,
        current_frame: int,
        max_age_frames: int = 90,
    ) -> List[int]:
        """
        Get track IDs that were lost within max_age_frames.

        Args:
            current_frame: Current frame number
            max_age_frames: Maximum age to consider

        Returns:
            List of lost track IDs
        """
        lost = []
        for track_id, history in self._track_embeddings.items():
            age = current_frame - history.last_frame
            if 0 < age <= max_age_frames:
                lost.append(track_id)

        return lost

    def attempt_reidentification(
        self,
        new_track_ids: List[int],
        current_frame: int,
        max_age_frames: int = 90,
    ) -> Dict[int, int]:
        """
        Attempt to re-identify new tracks with lost tracks.

        Args:
            new_track_ids: List of new track IDs
            current_frame: Current frame number
            max_age_frames: Max age for lost tracks

        Returns:
            Dict mapping new_track_id -> matched_lost_track_id
        """
        lost_tracks = self.get_lost_tracks(current_frame, max_age_frames)

        if not lost_tracks:
            return {}

        matches = {}

        for new_id in new_track_ids:
            match = self.find_best_match(new_id, lost_tracks)
            if match:
                matched_id, similarity = match
                matches[new_id] = matched_id

                # Remove from candidates to avoid duplicate matches
                lost_tracks.remove(matched_id)

        return matches

    def get_track_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get mean embedding for a track."""
        if track_id not in self._track_embeddings:
            return None
        return self._track_embeddings[track_id].get_mean_embedding()

    def get_similarity_matrix(
        self,
        track_ids: List[int],
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for tracks.

        Args:
            track_ids: List of track IDs

        Returns:
            NxN similarity matrix
        """
        n = len(track_ids)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(track_ids[i], track_ids[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    def reset(self) -> None:
        """Reset all track embeddings."""
        self._track_embeddings.clear()


def create_siglip_reid(cfg=None) -> SigLIPReID:
    """
    Create SigLIPReID from config.

    Args:
        cfg: CVConfig instance

    Returns:
        Configured SigLIPReID
    """
    if cfg is None:
        return SigLIPReID()

    reid = SigLIPReID(
        model_name=getattr(cfg, "siglip_model_name", "google/siglip-base-patch16-224"),
        similarity_threshold=getattr(cfg, "siglip_similarity_threshold", 0.85),
        embedding_dim=getattr(cfg, "siglip_embedding_dim", 768),
        crop_padding=getattr(cfg, "siglip_crop_padding", 0.1),
    )

    # Load model if re-ID enabled
    if getattr(cfg, "enable_siglip_reid", False):
        reid.load_model()

    return reid
