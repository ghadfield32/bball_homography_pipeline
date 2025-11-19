# api/src/cv/test_pipeline.py
"""
Test utilities for validating the CV pipeline components.

Run this script to verify that tracking, homography, and pose estimation
are working correctly.

Usage:
    python -m api.src.cv.test_pipeline

    # Or test individual components:
    python -m api.src.cv.test_pipeline --component tracker
    python -m api.src.cv.test_pipeline --component homography
    python -m api.src.cv.test_pipeline --component pose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import supervision as sv


def test_tracker_module() -> bool:
    """Test the PlayerTracker module."""
    print("\n=== Testing PlayerTracker Module ===")

    try:
        from api.src.cv.tracker import PlayerTracker, TrackState

        # Test TrackState
        track = TrackState(track_id=1)
        track.add_observation(
            frame_idx=0,
            bbox_image=np.array([100, 100, 200, 300]),
            point_image=np.array([150, 300]),
            point_court=np.array([10.0, 25.0]),
            team_id=0,
            confidence=0.9,
            fps=30.0
        )
        track.add_observation(
            frame_idx=1,
            bbox_image=np.array([105, 100, 205, 300]),
            point_image=np.array([155, 300]),
            point_court=np.array([11.0, 25.0]),
            team_id=0,
            confidence=0.85,
            fps=30.0
        )

        assert track.is_active, "Track should be active"
        assert track.last_frame == 1, f"Expected last_frame=1, got {track.last_frame}"
        assert track.total_distance_court_ft > 0, "Distance should be > 0"
        print(f"  TrackState: OK (distance={track.total_distance_court_ft:.2f}ft)")

        # Test PlayerTracker initialization
        tracker = PlayerTracker(
            track_activation_threshold=0.20,
            lost_track_buffer=60,
            minimum_matching_threshold=0.6,
            frame_rate=30
        )
        print(f"  PlayerTracker init: OK")
        print(f"    - track_activation_threshold: {tracker.track_activation_threshold}")
        print(f"    - lost_track_buffer: {tracker.lost_track_buffer}")
        print(f"    - minimum_matching_threshold: {tracker.minimum_matching_threshold}")

        # Test update with mock detections
        mock_xyxy = np.array([[100, 100, 200, 300], [300, 100, 400, 300]], dtype=float)
        mock_conf = np.array([0.9, 0.85])
        mock_class = np.array([0, 0])

        dets = sv.Detections(
            xyxy=mock_xyxy,
            confidence=mock_conf,
            class_id=mock_class
        )

        tracked = tracker.update(dets, frame_idx=0)
        print(f"  Tracker update: OK (tracked {len(tracked)} detections)")

        # Check analytics
        analytics = tracker.get_analytics()
        print(f"  Analytics: total_tracks={analytics['total_tracks']}, active={analytics['active_tracks']}")

        print("  [PASS] PlayerTracker module tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] PlayerTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_homography_calibrator() -> bool:
    """Test the HomographyCalibrator module."""
    print("\n=== Testing HomographyCalibrator Module ===")

    try:
        from api.src.cv.homography_calibrator import HomographyCalibrator, SegmentData
        from sports.basketball import CourtConfiguration, League
        from sports import MeasurementUnit

        # Create court config
        court_config = CourtConfiguration(
            league=League.NBA,
            measurement_unit=MeasurementUnit.FEET
        )

        # Test calibrator initialization
        calibrator = HomographyCalibrator(
            court_config=court_config,
            confidence_threshold=0.5,
            min_keypoints=4,
            enable_semantic_constraints=True
        )
        print(f"  HomographyCalibrator init: OK")
        print(f"    - enable_semantic_constraints: {calibrator.enable_semantic_constraints}")
        print(f"    - line_collinearity_threshold: {calibrator.line_collinearity_threshold}ft")
        print(f"    - arc_radius_threshold: {calibrator.arc_radius_threshold}ft")

        # Test SegmentData
        segment = SegmentData(segment_id=0, start_frame=0)
        assert not segment.is_calibrated, "New segment should not be calibrated"
        print(f"  SegmentData: OK")

        # Test semantic validation functions exist
        import cv2
        H_test = np.eye(3, dtype=np.float32)
        test_pts = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)

        valid, deviation = calibrator._validate_line_collinearity(H_test, test_pts)
        print(f"  Line collinearity validation: OK (valid={valid}, deviation={deviation:.3f}ft)")

        # Test quality mask
        mask = calibrator.get_quality_mask()
        print(f"  Quality mask: OK (empty={len(mask) == 0})")

        print("  [PASS] HomographyCalibrator module tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] HomographyCalibrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_pipeline() -> bool:
    """Test the PosePipeline module."""
    print("\n=== Testing PosePipeline Module ===")

    try:
        from api.src.cv.pose_pipeline import (
            PosePipeline, PoseObservation, PlayerPoseHistory,
            JointID, SKELETON_CONNECTIONS
        )

        # Test JointID enum
        assert JointID.NOSE == 0
        assert JointID.RIGHT_WRIST == 10
        print(f"  JointID enum: OK")

        # Test skeleton connections
        assert len(SKELETON_CONNECTIONS) > 0
        print(f"  Skeleton connections: OK ({len(SKELETON_CONNECTIONS)} connections)")

        # Test PoseObservation
        keypoints = np.random.rand(17, 2).astype(np.float32) * 100
        confidences = np.random.rand(17).astype(np.float32)

        obs = PoseObservation(
            frame_idx=0,
            track_id=1,
            keypoints_image=keypoints,
            confidences=confidences
        )
        print(f"  PoseObservation: OK (valid={obs.is_valid})")

        # Test PlayerPoseHistory
        history = PlayerPoseHistory(track_id=1)
        history.add_observation(obs)

        frames, positions = history.get_joint_trajectory(JointID.RIGHT_WRIST, as_court=False)
        print(f"  PlayerPoseHistory: OK (observations={len(history.observations)})")

        # Test PosePipeline initialization (without loading model)
        pipeline = PosePipeline(
            model_name="yolov8n-pose",
            confidence_threshold=0.3
        )
        print(f"  PosePipeline init: OK")
        print(f"    - model_name: {pipeline.model_name}")
        print(f"    - confidence_threshold: {pipeline.confidence_threshold}")

        # Test IoU computation
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        iou = pipeline._compute_iou(box1, box2)
        expected_iou = 2500 / (10000 + 10000 - 2500)  # intersection / union
        assert abs(iou - expected_iou) < 0.01, f"IoU mismatch: {iou} vs {expected_iou}"
        print(f"  IoU computation: OK (iou={iou:.3f})")

        print("  [PASS] PosePipeline module tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] PosePipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jersey_ocr() -> bool:
    """Test the JerseyOCR module."""
    print("\n=== Testing JerseyOCR Module ===")

    try:
        from api.src.cv.jersey_ocr import JerseyOCR, JerseyDetection, TrackNumberHistory

        # Test JerseyDetection
        detection = JerseyDetection(number="23", confidence=0.95)
        assert detection.number == "23"
        print(f"  JerseyDetection: OK")

        # Test TrackNumberHistory
        history = TrackNumberHistory(track_id=1)
        history.add_detection("23", confidence=0.9)
        history.add_detection("23", confidence=0.8)
        history.add_detection("32", confidence=0.5)  # Noise

        number = history.get_number()
        confidence = history.get_confidence()
        assert number == "23", f"Expected '23', got {number}"
        assert confidence > 0.5, f"Confidence should be > 0.5, got {confidence}"
        print(f"  TrackNumberHistory: OK (number={number}, conf={confidence:.2f})")

        # Test JerseyOCR initialization
        ocr = JerseyOCR(min_confidence=0.5)
        print(f"  JerseyOCR init: OK")
        print(f"    - min_confidence: {ocr.min_confidence}")

        # Test track number retrieval
        ocr._track_histories[1] = history
        retrieved = ocr.get_track_number(1)
        assert retrieved == "23"
        print(f"  Track number retrieval: OK")

        # Test find by number
        found = ocr.find_track_by_number("23")
        assert 1 in found
        print(f"  Find by number: OK (found tracks: {found})")

        print("  [PASS] JerseyOCR module tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] JerseyOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config() -> bool:
    """Test that config has all new parameters."""
    print("\n=== Testing CVConfig Parameters ===")

    try:
        from api.src.cv.config import CVConfig

        cfg = CVConfig()

        # Check tracking parameters
        assert hasattr(cfg, 'enable_tracking'), "Missing enable_tracking"
        assert hasattr(cfg, 'track_activation_threshold'), "Missing track_activation_threshold"
        assert hasattr(cfg, 'lost_track_buffer'), "Missing lost_track_buffer"
        assert hasattr(cfg, 'minimum_matching_threshold'), "Missing minimum_matching_threshold"
        assert hasattr(cfg, 'minimum_consecutive_frames'), "Missing minimum_consecutive_frames"
        print(f"  Tracking parameters: OK")
        print(f"    - track_activation_threshold: {cfg.track_activation_threshold}")
        print(f"    - lost_track_buffer: {cfg.lost_track_buffer}")

        # Check segment homography parameters
        assert hasattr(cfg, 'enable_segment_homography'), "Missing enable_segment_homography"
        assert hasattr(cfg, 'segment_min_frames'), "Missing segment_min_frames"
        assert hasattr(cfg, 'segment_change_threshold'), "Missing segment_change_threshold"
        print(f"  Segment homography parameters: OK")

        # Check pose parameters
        assert hasattr(cfg, 'enable_pose_estimation'), "Missing enable_pose_estimation"
        assert hasattr(cfg, 'pose_model_name'), "Missing pose_model_name"
        assert hasattr(cfg, 'video_fps'), "Missing video_fps"
        print(f"  Pose estimation parameters: OK")
        print(f"    - pose_model_name: {cfg.pose_model_name}")

        print("  [PASS] CVConfig tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] CVConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests() -> bool:
    """Run all component tests."""
    print("=" * 60)
    print("CV Pipeline Component Tests")
    print("=" * 60)

    results = {
        "config": test_config(),
        "tracker": test_tracker_module(),
        "homography": test_homography_calibrator(),
        "pose": test_pose_pipeline(),
        "jersey_ocr": test_jersey_ocr(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test CV pipeline components")
    parser.add_argument(
        "--component",
        choices=["all", "tracker", "homography", "pose", "jersey_ocr", "config"],
        default="all",
        help="Component to test"
    )
    args = parser.parse_args()

    if args.component == "all":
        success = run_all_tests()
    elif args.component == "tracker":
        success = test_tracker_module()
    elif args.component == "homography":
        success = test_homography_calibrator()
    elif args.component == "pose":
        success = test_pose_pipeline()
    elif args.component == "jersey_ocr":
        success = test_jersey_ocr()
    elif args.component == "config":
        success = test_config()
    else:
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
