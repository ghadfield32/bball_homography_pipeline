# api/src/cv/jersey_ocr.py
"""
Jersey number OCR module for basketball player re-identification.

Provides jersey number recognition with:
- Per-track number history with majority voting for stability
- Integration points for OCR models (EasyOCR, PaddleOCR, or custom)
- Re-identification across camera cuts using number + team color

This module provides the structure for jersey OCR. The actual OCR model
can be plugged in via the detect_numbers() method.

Usage:
    jersey_ocr = JerseyOCR()

    for frame_idx, frame in enumerate(video):
        tracked_dets = tracker.update(...)
        numbers = jersey_ocr.detect_and_update(frame, tracked_dets)

        # Get stable number for a track
        number = jersey_ocr.get_track_number(track_id)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


@dataclass
class JerseyDetection:
    """Single jersey number detection."""
    number: str  # Detected number as string (e.g., "23", "00")
    confidence: float
    bbox: Optional[np.ndarray] = None  # Crop region in image


@dataclass
class TrackNumberHistory:
    """Number detection history for a single track."""
    track_id: int

    # History of detections: {number_str: count}
    number_votes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Most recent detections for recency weighting
    recent_numbers: List[str] = field(default_factory=list)
    max_recent: int = 10

    # Confidence history
    total_detections: int = 0

    def add_detection(self, number: str, confidence: float = 1.0) -> None:
        """Add a number detection to history."""
        if not number:
            return

        # Add to vote count (weighted by confidence)
        weight = int(max(1, confidence * 3))  # 1-3 votes based on confidence
        self.number_votes[number] += weight

        # Add to recent list
        self.recent_numbers.append(number)
        if len(self.recent_numbers) > self.max_recent:
            self.recent_numbers.pop(0)

        self.total_detections += 1

    def get_number(self) -> Optional[str]:
        """Get most likely number using majority voting."""
        if not self.number_votes:
            return None

        # Return number with most votes
        return max(self.number_votes, key=self.number_votes.get)

    def get_confidence(self) -> float:
        """Get confidence in the current number assignment."""
        if not self.number_votes or self.total_detections == 0:
            return 0.0

        best_number = self.get_number()
        if best_number is None:
            return 0.0

        # Confidence = votes for best / total votes
        total_votes = sum(self.number_votes.values())
        return self.number_votes[best_number] / total_votes if total_votes > 0 else 0.0


@dataclass
class JerseyOCR:
    """
    Jersey number recognition for player re-identification.

    Maintains per-track number history with majority voting for stable
    number assignments even with noisy OCR.

    The actual OCR can be performed by:
    - EasyOCR
    - PaddleOCR
    - Custom digit recognition model
    - Roboflow jersey number model

    This class provides the tracking and voting logic.
    """
    # OCR parameters
    min_confidence: float = 0.5
    crop_padding: float = 0.1  # Padding around player bbox for number region

    # Number region (relative to bbox)
    number_region_top: float = 0.1  # Start at 10% from top
    number_region_bottom: float = 0.5  # End at 50% from top
    number_region_left: float = 0.2
    number_region_right: float = 0.8

    # Track histories
    _track_histories: Dict[int, TrackNumberHistory] = field(default_factory=dict)

    # OCR model (set via load_model)
    _ocr_model = None
    _ocr_type: str = "none"  # "easyocr", "paddleocr", "custom", "none"

    def __post_init__(self):
        """Initialize state."""
        self._track_histories = {}

    def load_model(self, model_type: str = "easyocr") -> bool:
        """
        Load OCR model.

        Args:
            model_type: Type of OCR to use ("easyocr", "paddleocr", "none")

        Returns:
            True if model loaded successfully
        """
        self._ocr_type = model_type

        if model_type == "easyocr":
            try:
                import easyocr
                self._ocr_model = easyocr.Reader(['en'], gpu=True)
                print(f"[INFO][jersey_ocr] Loaded EasyOCR")
                return True
            except ImportError:
                print(f"[WARN][jersey_ocr] EasyOCR not installed")
                return False
            except Exception as e:
                print(f"[WARN][jersey_ocr] Failed to load EasyOCR: {e}")
                return False

        elif model_type == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self._ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
                print(f"[INFO][jersey_ocr] Loaded PaddleOCR")
                return True
            except ImportError:
                print(f"[WARN][jersey_ocr] PaddleOCR not installed")
                return False
            except Exception as e:
                print(f"[WARN][jersey_ocr] Failed to load PaddleOCR: {e}")
                return False

        else:
            print(f"[INFO][jersey_ocr] No OCR model loaded (model_type={model_type})")
            return False

    def _extract_number_region(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract the jersey number region from a player bounding box.

        Args:
            frame: Full frame image
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            Cropped number region or None
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1

        # Extract number region (upper torso area)
        ny1 = y1 + int(h * self.number_region_top)
        ny2 = y1 + int(h * self.number_region_bottom)
        nx1 = x1 + int(w * self.number_region_left)
        nx2 = x1 + int(w * self.number_region_right)

        # Clamp to image bounds
        fh, fw = frame.shape[:2]
        ny1 = max(0, min(fh - 1, ny1))
        ny2 = max(0, min(fh, ny2))
        nx1 = max(0, min(fw - 1, nx1))
        nx2 = max(0, min(fw, nx2))

        if ny2 <= ny1 or nx2 <= nx1:
            return None

        return frame[ny1:ny2, nx1:nx2]

    def _run_ocr(self, crop: np.ndarray) -> List[JerseyDetection]:
        """
        Run OCR on a cropped region.

        Args:
            crop: Cropped image region

        Returns:
            List of detected numbers
        """
        if self._ocr_model is None:
            return []

        detections = []

        try:
            if self._ocr_type == "easyocr":
                results = self._ocr_model.readtext(crop, detail=1)
                for bbox, text, conf in results:
                    # Filter to digits only
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits and len(digits) <= 3 and conf >= self.min_confidence:
                        detections.append(JerseyDetection(
                            number=digits,
                            confidence=conf
                        ))

            elif self._ocr_type == "paddleocr":
                results = self._ocr_model.ocr(crop, cls=True)
                if results and results[0]:
                    for line in results[0]:
                        text, conf = line[1]
                        digits = ''.join(c for c in text if c.isdigit())
                        if digits and len(digits) <= 3 and conf >= self.min_confidence:
                            detections.append(JerseyDetection(
                                number=digits,
                                confidence=conf
                            ))

        except Exception as e:
            print(f"[WARN][jersey_ocr] OCR failed: {e}")

        return detections

    def detect_numbers(
        self, frame: np.ndarray, tracked_detections
    ) -> Dict[int, JerseyDetection]:
        """
        Detect jersey numbers for tracked players.

        Args:
            frame: Current frame
            tracked_detections: sv.Detections with tracker_id

        Returns:
            Dict mapping track_id -> JerseyDetection
        """
        results = {}

        if tracked_detections.tracker_id is None:
            return results

        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)
            bbox = tracked_detections.xyxy[i]

            # Extract number region
            crop = self._extract_number_region(frame, bbox)
            if crop is None:
                continue

            # Run OCR
            detections = self._run_ocr(crop)

            if detections:
                # Take highest confidence detection
                best = max(detections, key=lambda d: d.confidence)
                results[track_id] = best

        return results

    def detect_and_update(
        self, frame: np.ndarray, tracked_detections
    ) -> Dict[int, JerseyDetection]:
        """
        Detect numbers and update track histories.

        Args:
            frame: Current frame
            tracked_detections: sv.Detections with tracker_id

        Returns:
            Dict mapping track_id -> JerseyDetection
        """
        detections = self.detect_numbers(frame, tracked_detections)

        # Update histories
        for track_id, detection in detections.items():
            if track_id not in self._track_histories:
                self._track_histories[track_id] = TrackNumberHistory(track_id=track_id)

            self._track_histories[track_id].add_detection(
                detection.number, detection.confidence
            )

        return detections

    def get_track_number(self, track_id: int) -> Optional[str]:
        """Get the stable jersey number for a track."""
        history = self._track_histories.get(track_id)
        if history is None:
            return None
        return history.get_number()

    def get_track_confidence(self, track_id: int) -> float:
        """Get confidence in the track's number assignment."""
        history = self._track_histories.get(track_id)
        if history is None:
            return 0.0
        return history.get_confidence()

    def get_all_track_numbers(self) -> Dict[int, Tuple[str, float]]:
        """
        Get stable numbers for all tracks.

        Returns:
            Dict mapping track_id -> (number, confidence)
        """
        results = {}
        for track_id, history in self._track_histories.items():
            number = history.get_number()
            if number:
                results[track_id] = (number, history.get_confidence())
        return results

    def find_track_by_number(self, number: str) -> List[int]:
        """
        Find track IDs that have a specific jersey number.

        Args:
            number: Jersey number to search for

        Returns:
            List of track IDs with that number
        """
        return [
            track_id for track_id, history in self._track_histories.items()
            if history.get_number() == number
        ]

    def merge_tracks(self, old_track_id: int, new_track_id: int) -> None:
        """
        Merge track histories when re-identifying a player.

        Used when a track dies and is later re-identified as the same player.

        Args:
            old_track_id: Previous track ID
            new_track_id: New track ID to merge into
        """
        if old_track_id not in self._track_histories:
            return

        old_history = self._track_histories[old_track_id]

        if new_track_id not in self._track_histories:
            self._track_histories[new_track_id] = TrackNumberHistory(track_id=new_track_id)

        new_history = self._track_histories[new_track_id]

        # Merge vote counts
        for number, votes in old_history.number_votes.items():
            new_history.number_votes[number] += votes

        # Merge recent lists
        new_history.recent_numbers = old_history.recent_numbers + new_history.recent_numbers
        new_history.recent_numbers = new_history.recent_numbers[-new_history.max_recent:]

        new_history.total_detections += old_history.total_detections

    def reset(self) -> None:
        """Reset all track histories."""
        self._track_histories.clear()


def create_jersey_ocr(cfg=None, model_type: str = "none") -> JerseyOCR:
    """
    Create JerseyOCR from config.

    Args:
        cfg: CVConfig instance
        model_type: OCR model type ("easyocr", "paddleocr", "none")

    Returns:
        Configured JerseyOCR
    """
    ocr = JerseyOCR()

    if model_type != "none":
        ocr.load_model(model_type)

    return ocr
