#!/usr/bin/env python3
"""
Computer Vision validation and testing for Basketball Detection setup.
Tests YOLO, Roboflow, OpenCV, and tracking libraries.
"""
import sys
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_opencv() -> bool:
    """Test OpenCV installation and basic functionality."""
    print_section("OPENCV TEST")
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Build info: {cv2.getBuildInformation().split('General configuration')[0][:100]}...")
        
        # Check available video backends
        backends = []
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_V4L2]:
            try:
                cap = cv2.VideoCapture()
                if cap.open(0, backend):
                    backends.append(backend)
                    cap.release()
            except:
                pass
        
        print(f"Available video backends: {backends}")
        
        # Test basic image operations
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        print(f"Image processing test: OK (edges shape: {edges.shape})")
        return True
        
    except Exception as e:
        print(f"OpenCV test failed: {e}")
        return False


def test_ultralytics() -> bool:
    """Test Ultralytics YOLO installation and model loading."""
    print_section("ULTRALYTICS YOLO TEST")
    try:
        from ultralytics import YOLO, __version__
        from ultralytics.utils.checks import check_requirements
        
        print(f"Ultralytics version: {__version__}")
        
        # Check if CUDA is available for YOLO
        import torch
        print(f"YOLO CUDA available: {torch.cuda.is_available()}")
        
        # Test loading a model (will download if not cached)
        model_path = "/app/weights/yolov8n.pt"
        if not os.path.exists(model_path):
            print("Downloading YOLOv8n model...")
            model = YOLO("yolov8n.pt")
        else:
            print(f"Loading cached model from {model_path}")
            model = YOLO(model_path)
        
        print(f"Model loaded: {model.model.__class__.__name__}")
        print(f"Model device: {model.device}")
        
        # Test inference on dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        
        print(f"Inference test: OK (processed {len(results)} image)")
        return True
        
    except Exception as e:
        print(f"Ultralytics test failed: {e}")
        return False


def test_roboflow() -> bool:
    """Test Roboflow installation and API connectivity."""
    print_section("ROBOFLOW TEST")
    try:
        import roboflow
        from roboflow import Roboflow
        
        print(f"Roboflow version: {roboflow.__version__}")
        
        # Check if API key is set
        api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        if api_key:
            print("Roboflow API key: Set")
            
            # Test API connection (won't actually download without valid key)
            try:
                rf = Roboflow(api_key=api_key)
                print("Roboflow client initialized successfully")
                
                # Try to access basketball court detection project
                workspace = os.environ.get('ROBOFLOW_WORKSPACE', 'basketball-formations')
                project = os.environ.get('ROBOFLOW_PROJECT', 'basketball-court-detection-2-mlopt')
                version = os.environ.get('ROBOFLOW_VERSION', '1')
                
                print(f"Workspace: {workspace}")
                print(f"Project: {project}")
                print(f"Version: {version}")
                
                # Note: This will fail without valid API key, which is expected
                # project = rf.workspace(workspace).project(project)
                # dataset = project.version(version)
                # print(f"Dataset accessible: {dataset.name}")
                
            except Exception as e:
                print(f"Note: Full Roboflow test requires valid API key: {e}")
        else:
            print("Roboflow API key: Not set (set ROBOFLOW_API_KEY to enable)")
            print("Get your API key from: https://app.roboflow.com/settings/api")
        
        return True
        
    except Exception as e:
        print(f"Roboflow test failed: {e}")
        return False


def test_supervision() -> bool:
    """Test supervision library for tracking and annotation."""
    print_section("SUPERVISION TRACKING TEST")
    try:
        import supervision as sv
        
        print(f"Supervision version: {sv.__version__}")
        
        # Test basic detection utilities
        from supervision import Detections, BoxAnnotator
        
        # Create dummy detections
        xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        confidence = np.array([0.9, 0.8])
        class_id = np.array([0, 1])
        
        detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        print(f"Created {len(detections)} dummy detections")
        
        # Test annotator
        annotator = BoxAnnotator()
        print("Box annotator initialized")
        
        # Test tracker availability
        try:
            from supervision import ByteTrack
            tracker = ByteTrack()
            print("ByteTrack tracker available")
        except ImportError:
            print("ByteTrack tracker not available (optional)")
        
        return True
        
    except Exception as e:
        print(f"Supervision test failed: {e}")
        return False


def test_video_processing() -> bool:
    """Test video processing capabilities."""
    print_section("VIDEO PROCESSING TEST")
    try:
        # Test ffmpeg-python
        import ffmpeg
        print("ffmpeg-python: Available")
        
        # Check ffmpeg binary
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_version = result.stdout.split('\n')[0]
            print(f"ffmpeg binary: {ffmpeg_version}")
        else:
            print("ffmpeg binary: Not found or error")
            
        # Test moviepy
        try:
            import moviepy
            print(f"MoviePy version: {moviepy.__version__}")
        except Exception as e:
            print(f"MoviePy: {e}")
            
        # Test yt-dlp
        try:
            import yt_dlp
            print(f"yt-dlp version: {yt_dlp.version.__version__}")
        except Exception as e:
            print(f"yt-dlp: {e}")
            
        return True
        
    except Exception as e:
        print(f"Video processing test failed: {e}")
        return False


def test_basketball_models() -> bool:
    """Test basketball-specific model configurations."""
    print_section("BASKETBALL MODELS CONFIGURATION")
    
    print("Basketball Detection Models:")
    print("1. Court Detection: roboflow/basketball-court-detection-2")
    print("2. Player Detection: YOLOv8 (person class)")
    print("3. Ball Detection: Custom YOLO model")
    print("4. Jersey Number OCR: Available via Roboflow")
    
    # Check model directories
    model_dirs = ['/app/models', '/app/weights', '/workspace/models']
    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            if files:
                print(f"\n{dir_path}: {len(files)} files")
                for f in files[:5]:  # Show first 5 files
                    print(f"  - {f}")
            else:
                print(f"\n{dir_path}: Empty")
        else:
            print(f"\n{dir_path}: Not created yet")
    
    return True


def main() -> int:
    """Run all computer vision tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Computer Vision components")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        print_section("ENVIRONMENT VARIABLES")
        cv_vars = ['ROBOFLOW_API_KEY', 'YOLO_VERBOSE', 'OPENCV_LOG_LEVEL', 
                   'VIDEO_INPUT_DIR', 'VIDEO_OUTPUT_DIR', 'DISPLAY']
        for var in cv_vars:
            value = os.environ.get(var, 'Not set')
            if var == 'ROBOFLOW_API_KEY' and value != 'Not set':
                value = value[:10] + '...' if len(value) > 10 else value
            print(f"{var}: {value}")
    
    # Run tests
    opencv_ok = test_opencv()
    yolo_ok = test_ultralytics()
    
    if args.quick:
        print_section("QUICK TEST SUMMARY")
        print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
        print(f"YOLO: {'✓' if yolo_ok else '✗'}")
        return 0 if (opencv_ok and yolo_ok) else 1
    
    roboflow_ok = test_roboflow()
    supervision_ok = test_supervision()
    video_ok = test_video_processing()
    basketball_ok = test_basketball_models()
    
    print_section("COMPUTER VISION TEST SUMMARY")
    results = {
        "OpenCV": opencv_ok,
        "YOLO/Ultralytics": yolo_ok,
        "Roboflow": roboflow_ok,
        "Supervision": supervision_ok,
        "Video Processing": video_ok,
        "Basketball Config": basketball_ok
    }
    
    for component, status in results.items():
        status_symbol = "✓" if status else "✗"
        print(f"{component}: {status_symbol}")
    
    all_ok = all(results.values())
    
    if not all_ok:
        print("\n⚠️  Some components failed. Check the logs above for details.")
        print("Common fixes:")
        print("  - Set ROBOFLOW_API_KEY for Roboflow integration")
        print("  - Ensure ffmpeg is installed for video processing")
        print("  - Check GPU drivers for YOLO acceleration")
    else:
        print("\n✅ All computer vision components are working correctly!")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
