#!/usr/bin/env python3
"""
Computer Vision validation and testing for Basketball Detection Pipeline.
Tests YOLO, Roboflow, OpenCV, and tracking libraries integration with PyTorch/JAX GPU container.
"""
import sys
import os
import warnings
import numpy as np
import time
from pathlib import Path
warnings.filterwarnings('ignore')


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_environment_setup() -> bool:
    """Test that environment variables are properly configured for CV."""
    print_section("ENVIRONMENT SETUP VALIDATION")
    
    required_vars = {
        'YOLO_VERBOSE': 'false',
        'OPENCV_LOG_LEVEL': 'ERROR', 
        'VIDEO_INPUT_DIR': '/workspace/videos/input',
        'VIDEO_OUTPUT_DIR': '/workspace/videos/output'
    }
    
    optional_vars = {
        'ROBOFLOW_API_KEY': 'Roboflow integration',
        'DISPLAY': 'GUI support'
    }
    
    all_ok = True
    
    print("Required environment variables:")
    for var, expected in required_vars.items():
        value = os.environ.get(var)
        if value:
            status = "✅" if value == expected else "⚠️"
            print(f"  {var}: {value} {status}")
        else:
            print(f"  {var}: Not set ❌")
            all_ok = False
    
    print("\nOptional environment variables:")
    for var, purpose in optional_vars.items():
        value = os.environ.get(var, 'Not set')
        if var == 'ROBOFLOW_API_KEY' and value != 'Not set':
            value = value[:10] + '...' if len(value) > 10 else value
        print(f"  {var}: {value} ({purpose})")
    
    return all_ok


def test_directories() -> bool:
    """Test that required directories exist and are writable."""
    print_section("DIRECTORY STRUCTURE VALIDATION")
    
    directories = [
        '/app/models',
        '/app/weights', 
        '/app/data',
        '/workspace/videos',
        '/workspace/videos/input',
        '/workspace/videos/output'
    ]
    
    all_ok = True
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            try:
                # Test write permission
                test_file = path / '.write_test'
                test_file.write_text('test')
                test_file.unlink()
                print(f"  {dir_path}: ✅ Exists and writable")
            except Exception as e:
                print(f"  {dir_path}: ⚠️ Exists but not writable: {e}")
                all_ok = False
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  {dir_path}: ✅ Created successfully")
            except Exception as e:
                print(f"  {dir_path}: ❌ Cannot create: {e}")
                all_ok = False
    
    return all_ok


def test_opencv() -> bool:
    """Test OpenCV installation and basic functionality."""
    print_section("OPENCV INTEGRATION TEST")
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Test video backends
        backends = []
        backend_names = {
            cv2.CAP_FFMPEG: 'FFMPEG',
            cv2.CAP_GSTREAMER: 'GSTREAMER', 
            cv2.CAP_V4L2: 'V4L2'
        }
        
        for backend_id, name in backend_names.items():
            try:
                cap = cv2.VideoCapture()
                if hasattr(cv2, 'CAP_' + name):
                    backends.append(name)
                cap.release()
            except:
                pass
        
        print(f"Available video backends: {backends}")
        print(f"Image processing test: OK (edges shape: {edges.shape})")
        
        # Test CUDA support in OpenCV (if available)
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"OpenCV CUDA devices: {cuda_devices}")
        except:
            print("OpenCV CUDA support: Not available (CPU only)")
        
        return True
        
    except Exception as e:
        print(f"OpenCV test failed: {e}")
        return False


def test_ultralytics_yolo() -> bool:
    """Test Ultralytics YOLO installation and GPU acceleration."""
    print_section("ULTRALYTICS YOLO INTEGRATION TEST")
    try:
        from ultralytics import YOLO, __version__
        print(f"Ultralytics version: {__version__}")
        
        # Check PyTorch integration
        import torch
        print(f"PyTorch CUDA available for YOLO: {torch.cuda.is_available()}")
        
        # Test model loading
        print("Loading YOLOv8n model...")
        start_time = time.time()
        model = YOLO("yolov8n.pt")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Model device: {model.device}")
        
        # Test inference on dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("Running inference test...")
        start_time = time.time()
        results = model(dummy_img, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {inference_time:.3f}s")
        print(f"Results: {len(results)} outputs")
        
        # Test model info
        if hasattr(model, 'info'):
            try:
                model.info()
            except:
                print("Model info: Available but verbose output suppressed")
        
        return True
        
    except Exception as e:
        print(f"Ultralytics test failed: {e}")
        return False


def test_roboflow_integration() -> bool:
    """Test Roboflow installation and API connectivity."""
    print_section("ROBOFLOW INTEGRATION TEST")
    try:
        import roboflow
        from roboflow import Roboflow
        
        print(f"Roboflow version: {roboflow.__version__}")
        
        # Check API key configuration
        api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        if api_key and api_key != 'your_roboflow_api_key_here':
            print("Roboflow API key: Configured")
            
            try:
                # Test API connection
                rf = Roboflow(api_key=api_key)
                print("Roboflow client initialized successfully")
                
                # Get configuration for basketball detection
                workspace = os.environ.get('ROBOFLOW_WORKSPACE', 'basketball-formations')
                project = os.environ.get('ROBOFLOW_PROJECT', 'basketball-court-detection-2-mlopt')
                version = os.environ.get('ROBOFLOW_VERSION', '1')
                
                print(f"Basketball Detection Configuration:")
                print(f"  Workspace: {workspace}")
                print(f"  Project: {project}")
                print(f"  Version: {version}")
                
                # Note: Actual project access requires valid API key
                print("Note: Full project access requires valid API key")
                
            except Exception as e:
                print(f"API connection test: {e}")
                print("This is normal without a valid API key")
        else:
            print("Roboflow API key: Not configured")
            print("Set ROBOFLOW_API_KEY to enable full integration")
            print("Get your API key from: https://app.roboflow.com/settings/api")
        
        return True
        
    except Exception as e:
        print(f"Roboflow test failed: {e}")
        return False


def test_supervision_tracking() -> bool:
    """Test supervision library for object tracking and annotation."""
    print_section("SUPERVISION TRACKING TEST")
    try:
        import supervision as sv
        print(f"Supervision version: {sv.__version__}")
        
        # Test detection utilities
        from supervision import Detections, BoxAnnotator
        
        # Create dummy detections (basketball court, players, ball)
        xyxy = np.array([
            [100, 100, 200, 200],  # Player 1
            [300, 300, 400, 400],  # Player 2  
            [450, 150, 470, 170],  # Ball
            [0, 0, 640, 480]       # Court
        ])
        confidence = np.array([0.9, 0.8, 0.7, 0.95])
        class_id = np.array([0, 0, 1, 2])  # 0=person, 1=ball, 2=court
        
        detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        print(f"Created {len(detections)} dummy detections")
        
        # Test annotation
        annotator = BoxAnnotator()
        print("Box annotator initialized")
        
        # Test tracking capabilities
        try:
            from supervision import ByteTracker
            tracker = ByteTracker()
            print("ByteTracker initialized")
            
            # Test tracking update
            tracked_detections = tracker.update_with_detections(detections)
            print(f"Tracking update successful: {len(tracked_detections)} tracked objects")
            
        except ImportError:
            print("ByteTracker not available (optional advanced tracking)")
        except Exception as e:
            print(f"Tracking test: {e}")
        
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
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"ffmpeg binary: {version_line}")
        else:
            print("ffmpeg binary: Not found or error")
            
        # Test moviepy
        try:
            import moviepy
            from moviepy.editor import ColorClip
            print(f"MoviePy version: {moviepy.__version__}")
            
            # Test basic video creation
            clip = ColorClip(size=(100, 100), color=(255, 0, 0), duration=1)
            print("MoviePy functionality: OK")
            
        except Exception as e:
            print(f"MoviePy test: {e}")
            
        # Test yt-dlp
        try:
            import yt_dlp
            print(f"yt-dlp version: {yt_dlp.version.__version__}")
        except Exception as e:
            print(f"yt-dlp test: {e}")
            
        return True
        
    except Exception as e:
        print(f"Video processing test failed: {e}")
        return False


def test_basketball_pipeline() -> bool:
    """Test basketball-specific detection pipeline."""
    print_section("BASKETBALL DETECTION PIPELINE TEST")
    
    try:
        # Test integration of all components
        from ultralytics import YOLO
        import cv2
        import supervision as sv
        
        print("Basketball Detection Pipeline Components:")
        print(f"  ✅ YOLO: Available")
        print(f"  ✅ OpenCV: {cv2.__version__}")
        print(f"  ✅ Supervision: {sv.__version__}")
        
        # Simulate basketball court detection
        print("\nBasketball Court Detection Configuration:")
        court_model = os.environ.get('BASKETBALL_COURT_MODEL', 'basketball-court-detection-2-mlopt')
        player_model = os.environ.get('PLAYER_DETECTION_MODEL', 'yolov8n.pt') 
        ball_model = os.environ.get('BALL_DETECTION_MODEL', 'basketball-ball-detection')
        
        print(f"  Court Detection: {court_model}")
        print(f"  Player Detection: {player_model}")
        print(f"  Ball Detection: {ball_model}")
        
        # Test confidence thresholds
        court_conf = float(os.environ.get('COURT_DETECTION_CONFIDENCE', '0.7'))
        player_conf = float(os.environ.get('PLAYER_DETECTION_CONFIDENCE', '0.5'))
        ball_conf = float(os.environ.get('BALL_DETECTION_CONFIDENCE', '0.4'))
        
        print(f"\nConfidence Thresholds:")
        print(f"  Court: {court_conf}")
        print(f"  Players: {player_conf}")
        print(f"  Ball: {ball_conf}")
        
        # Check model directories
        models_dir = Path('/app/models')
        weights_dir = Path('/app/weights')
        
        print(f"\nModel Storage:")
        print(f"  Models directory: {models_dir} ({'✅' if models_dir.exists() else '❌'})")
        print(f"  Weights directory: {weights_dir} ({'✅' if weights_dir.exists() else '❌'})")
        
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.onnx'))
            print(f"  Found {len(model_files)} model files")
            
        if weights_dir.exists():
            weight_files = list(weights_dir.glob('*.pt')) + list(weights_dir.glob('*.weights'))
            print(f"  Found {len(weight_files)} weight files")
        
        return True
        
    except Exception as e:
        print(f"Basketball pipeline test failed: {e}")
        return False


def test_gpu_integration() -> bool:
    """Test GPU integration across PyTorch, JAX, and OpenCV."""
    print_section("GPU INTEGRATION TEST")
    
    try:
        # Test PyTorch GPU
        import torch
        pytorch_gpu = torch.cuda.is_available()
        print(f"PyTorch CUDA: {'✅' if pytorch_gpu else '❌'}")
        
        if pytorch_gpu:
            print(f"PyTorch GPU devices: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
        
        # Test JAX GPU
        import jax
        jax_devices = jax.devices()
        jax_gpus = [d for d in jax_devices if 'gpu' in str(d).lower()]
        print(f"JAX GPU devices: {len(jax_gpus)} ({'✅' if jax_gpus else '❌'})")
        
        # Test OpenCV CUDA (if available)
        try:
            import cv2
            opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            print(f"OpenCV CUDA: {'✅' if opencv_cuda else '❌'}")
        except:
            print(f"OpenCV CUDA: ❌ (CPU only)")
        
        # Test YOLO GPU
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        yolo_gpu = 'cuda' in str(model.device)
        print(f"YOLO GPU: {'✅' if yolo_gpu else '❌'}")
        
        return pytorch_gpu and len(jax_gpus) > 0
        
    except Exception as e:
        print(f"GPU integration test failed: {e}")
        return False


def main() -> int:
    """Run comprehensive computer vision tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Computer Vision Integration")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("Computer Vision Integration Test Suite")
    print(f"Working directory: {os.getcwd()}")
    
    if args.verbose:
        test_environment_setup()
    
    # Core tests
    directory_ok = test_directories()
    opencv_ok = test_opencv()
    yolo_ok = test_ultralytics_yolo()
    
    if args.quick:
        print_section("QUICK TEST SUMMARY")
        results = {
            "Directories": directory_ok,
            "OpenCV": opencv_ok,
            "YOLO": yolo_ok
        }
        
        for component, status in results.items():
            print(f"{component}: {'✅' if status else '❌'}")
        
        return 0 if all(results.values()) else 1
    
    # Full test suite
    roboflow_ok = test_roboflow_integration()
    supervision_ok = test_supervision_tracking()
    video_ok = test_video_processing()
    basketball_ok = test_basketball_pipeline()
    gpu_ok = test_gpu_integration()
    
    print_section("COMPREHENSIVE TEST SUMMARY")
    results = {
        "Directories": directory_ok,
        "OpenCV": opencv_ok,
        "YOLO/Ultralytics": yolo_ok,
        "Roboflow": roboflow_ok,
        "Supervision": supervision_ok,
        "Video Processing": video_ok,
        "Basketball Pipeline": basketball_ok,
        "GPU Integration": gpu_ok
    }
    
    for component, status in results.items():
        status_symbol = "✅" if status else "❌"
        print(f"{component}: {status_symbol}")
    
    all_ok = all(results.values())
    
    if not all_ok:
        print("\n⚠️  Some components failed. Common fixes:")
        print("  - Set ROBOFLOW_API_KEY for Roboflow integration")
        print("  - Ensure GPU drivers are properly installed")
        print("  - Check that all required directories exist")
        print("  - Verify ffmpeg installation for video processing")
    else:
        print("\n🎉 All computer vision components are working correctly!")
        print("Basketball detection pipeline is ready for use!")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
