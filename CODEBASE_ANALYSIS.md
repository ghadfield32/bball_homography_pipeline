# Basketball Homography Pipeline - Comprehensive Codebase Analysis

## Overview
The **bball_homography_pipeline** is a full-stack machine learning and computer vision project that combines:
- **FastAPI backend** for ML model serving (Bayesian inference, Random Forest, Logistic Regression)
- **React/Vite frontend** for UI and visualization
- **Advanced Computer Vision pipeline** for basketball court detection, player tracking, and homography transformation
- **MLflow integration** for model versioning and MLOps workflow
- **Docker/devcontainer setup** with GPU support (CUDA 12.4, JAX, PyTorch)

---

## Project Structure

```
bball_homography_pipeline/
├── api/                          # FastAPI Backend
│   ├── app/                       # Main application code
│   │   ├── main.py               # FastAPI app definition & routes
│   │   ├── models.py             # SQLAlchemy ORM models
│   │   ├── db.py                 # Database setup & lifespan
│   │   ├── security.py           # JWT/OAuth authentication
│   │   ├── crud.py               # Database CRUD operations
│   │   ├── core/
│   │   │   ├── config.py         # Configuration management (YAML + env)
│   │   │   └── env_utils.py      # Environment setup utilities
│   │   ├── ml/                    # ML backend setup
│   │   │   ├── builtin_trainers.py  # Model training (Iris, Cancer)
│   │   │   ├── pytensor_bootstrap.py # PyTensor/JAX initialization
│   │   │   └── utils.py          # ML utility functions
│   │   ├── schemas/              # Pydantic request/response models
│   │   │   ├── iris.py           # Iris species classification
│   │   │   ├── cancer.py         # Breast cancer diagnosis
│   │   │   ├── bayes.py          # Bayesian model parameters
│   │   │   └── train.py          # Training request schemas
│   │   ├── services/
│   │   │   └── ml/
│   │   │       └── model_service.py  # Model loading, training, serving
│   │   ├── middleware/           # HTTP middleware
│   │   │   └── concurrency.py    # Concurrent request limiting
│   │   └── deps/
│   │       └── limits.py         # Rate limiting dependencies
│   ├── src/                      # Computer Vision & Data modules
│   │   ├── cv/                   # Computer Vision Pipeline
│   │   │   ├── config.py         # CV configuration (models, paths, thresholds)
│   │   │   ├── homography.py     # Homography computation (OpenCV)
│   │   │   ├── fixed_homography.py  # Enhanced homography with proper keypoint mapping
│   │   │   ├── shot_tracker.py   # Shot event detection
│   │   │   ├── shot_pipeline.py  # Main CV pipeline (3338 lines)
│   │   │   ├── diagnostics/      # Diagnostic utilities
│   │   │   └── pipelines/        # Sub-pipelines
│   │   │       ├── homography_image.py  # Single image homography
│   │   │       └── video_pipeline.py    # Video processing
│   │   ├── trainers/             # Model training modules
│   │   │   ├── base.py           # Base trainer class
│   │   │   └── iris_rf_trainer.py  # Iris Random Forest trainer
│   │   ├── registry/             # Model registry
│   │   │   ├── types.py          # Registry type definitions
│   │   │   └── registry.py       # Model registry management
│   │   ├── airflow_project/      # Airflow DAGs (optional)
│   │   └── data_engineering.py   # Data preprocessing
│   ├── tests/                    # Backend tests
│   └── scripts/                  # Utility scripts
│       ├── seed_user.py          # Database seeding
│       ├── ensure_models.py      # Model initialization
│       └── promote.py            # Model promotion
├── web/                          # React Frontend
│   ├── src/
│   │   ├── App.jsx               # Main app component
│   │   ├── main.tsx              # Entry point
│   │   ├── components/           # React components
│   │   │   ├── Login.jsx         # Authentication
│   │   │   ├── Layout.jsx        # Main layout
│   │   │   ├── CancerForm.jsx    # Cancer prediction form
│   │   │   ├── IrisForm.jsx      # Iris prediction form
│   │   │   ├── ModelTraining.jsx # Training interface
│   │   │   ├── ResultsDisplay.jsx # Results visualization
│   │   │   └── MLModelFrontend.jsx  # Model status dashboard
│   │   └── services/             # API client services
│   └── config.yaml               # Frontend configuration
├── notebooks/                    # Jupyter notebooks
│   ├── backend/                  # Backend development notebooks
│   │   └── data_engineering/     # CV pipeline development
│   ├── 4090_gpu/                 # RTX 4090 specific notebooks
│   └── 5080_gpu/                 # RTX 5080 specific notebooks
├── .devcontainer/                # Development container
│   ├── Dockerfile                # Container image definition
│   ├── docker-compose.yml        # Multi-container setup
│   ├── devcontainer.json         # VS Code devcontainer config
│   └── tests/                    # Container validation tests
├── data/                         # Data storage
│   ├── quality_reports/          # Model quality reports
│   └── videos/                   # Video data
├── scripts/                      # Build & automation scripts
├── config.yaml                   # Global configuration
├── logging.yaml                  # Logging configuration
├── pyproject.toml                # Python dependencies
├── docker-compose.yml            # Production container setup
└── README.md                     # Documentation
```

---

## Core Components

### 1. FastAPI Backend (`/api/app`)

#### Main Application (`main.py`)
- **Framework**: FastAPI with async/await
- **Authentication**: JWT-based OAuth2
- **Rate Limiting**: Redis-backed token bucket with configurable limits per endpoint
- **Prediction Caching**: Redis cache for ML predictions with TTL
- **Concurrency Control**: Request throttling middleware
- **MLflow Integration**: Model versioning and lifecycle management

#### Key Endpoints

**Health & Readiness**:
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/ready` - App readiness status
- `GET /api/v1/ready/frontend` - Frontend-safe readiness data
- `GET /api/v1/ready/full` - Detailed readiness with env drift detection

**Authentication**:
- `POST /api/v1/token` - JWT token issuance (rate limited)
- `GET /api/v1/hello` - Authenticated echo endpoint

**Predictions**:
- `POST /api/v1/iris/predict` - Iris species classification (RF or LogReg)
- `POST /api/v1/cancer/predict` - Breast cancer diagnosis (Bayesian, LogReg, or RF)
  - Optional: posterior uncertainty sampling
  - Optional: Redis caching

**Model Training**:
- `POST /api/v1/iris/train` - Kick off iris model training (async)
- `POST /api/v1/cancer/train` - Kick off cancer model training (async)
- `POST /api/v1/cancer/bayes/train` - Bayesian cancer training with hyperparameters
- `GET /api/v1/cancer/bayes/config` - Bayesian training config template

**MLOps**:
- `POST /api/v1/mlops/evaluate/{model_name}` - Quality gate evaluation
- `POST /api/v1/mlops/promote/{model_name}/staging` - Promote to staging
- `POST /api/v1/mlops/promote/{model_name}/production` - Promote to production
- `POST /api/v1/mlops/reload-model` - Hot-reload models
- `GET /api/v1/mlops/status` - MLOps status dashboard
- `GET /api/v1/mlops/models/{model_name}/metrics` - Model version metrics
- `GET /api/v1/mlops/models/{model_name}/compare` - Compare model versions
- `GET /api/v1/mlops/models/{model_name}/quality-gate` - Check quality gate

**Debug Endpoints**:
- `GET /api/v1/debug/ready` - Configuration verification
- `GET /api/v1/debug/effective-config` - Effective runtime config (with redaction)
- `GET /api/v1/debug/compiler` - JAX/NumPyro backend info
- `GET /api/v1/debug/psutil` - Process utilities status
- `GET /api/v1/debug/deps` - Dependency version audit
- `GET /api/v1/debug/ratelimit/{bucket}` - Rate limit inspection
- `POST /api/v1/debug/ratelimit/reset` - Reset rate limit counters

#### Database Models
- **User**: Username/password for authentication (SQLAlchemy ORM)
- Database: SQLite (dev) or PostgreSQL (production-ready)

#### Security
- **JWT Tokens**: 30-minute expiration (configurable)
- **Password Hashing**: bcrypt
- **OAuth2 PasswordBearer**: Standard FastAPI integration
- **CORS**: Configurable origins

### 2. Machine Learning Backend

#### Model Service (`services/ml/model_service.py`)
Manages the complete ML lifecycle:
- **Model Loading**: Async model initialization from MLflow registry
- **Model Serving**: Real-time predictions with caching
- **Training**: Background task training with MLflow logging
- **Quality Gates**: Accuracy/F1 thresholds for promotion
- **Dependency Auditing**: Detects package version mismatches
- **Self-Healing**: Auto-trains missing models in background

#### Models Supported
1. **Iris Species Classification**
   - Random Forest (Default)
   - Logistic Regression
   - Dataset: 150 samples, 4 features, 3 species

2. **Breast Cancer Diagnosis**
   - Bayesian Hierarchical Model (JAX/NumPyro) - Primary
   - Random Forest
   - Logistic Regression
   - Dataset: 569 samples, 30 features
   - Output: Malignant/Benign with uncertainty estimates

#### ML Stack
- **Core**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Bayesian**: PyMC 5.20+, NumPyro, JAX, PyTensor
- **Model Registry**: MLflow with SQLite/PostgreSQL backend
- **Experiment Tracking**: MLflow UI on port 5000

#### Training Configuration
- **Bayesian Sampling**: NUTS sampler via NumPyro
- **Hyperparameter Optimization**: Optuna with MLflow integration
- **Model Versioning**: Automatic with stage promotion (Dev → Staging → Production)
- **Garbage Collection**: Keeps latest N runs per model (configurable)

### 3. Computer Vision Pipeline

#### Architecture
```
Input Video
    ↓
[Frame Extraction]
    ↓
[Player Detection] ← YOLO v8 (basketball-player-detection-3)
[Court Keypoint Detection] ← Custom Roboflow model (basketball-court-detection-2)
    ↓
[Team Classification] ← Torso HSV color clustering (k-means)
    ↓
[Homography Computation] ← RANSAC-based court transformation
    ↓
[Shot Event Tracking] ← Jump shot/Layup/Dunk/Ball-in-basket detection
    ↓
[Annotation & Rendering] ← Court map overlay + broadcast visualization
    ↓
Output Videos + Analytics
```

#### Key Modules

**Config (`config.py`)** - 192 lines
- Model IDs: Player & court detection models
- Thresholds: Confidence, IOU, keypoint matching
- Paths: Video/image input/output directories
- Court drawing: NBA court configuration
- Event logic: Reset times, cooldown periods
- Roboflow API configuration

**Homography Computation** - 400 lines each
- `homography.py`: OpenCV perspective transform
- `fixed_homography.py`: Enhanced version with proper keypoint-to-vertex mapping
  - KeypointMapping class: Maps 30 court keypoints to court vertices
  - RANSAC: Robust outlier rejection
  - Quality validation: RMSE & max error thresholds
  - Debug utilities: Detailed correspondence logging

**Shot Event Tracking (`shot_tracker.py`)** - 207 lines
- Event types: START, MADE, MISSED
- Detection logic:
  - Jump shot: Consecutive frames with jump shot detection
  - Layup/Dunk: Consecutive frames with layup/dunk detection
  - Made: Ball-in-basket detection
  - Reset: Timeout reset period
- Cooldown handling: Prevents duplicate shot events

**Main Pipeline (`shot_pipeline.py`)** - 3338 lines
- Player detection & grouping:
  - Team separation via torso HSV clustering
  - Referee identification by class/label
- Homography computation per frame
- Shot event tracking
- Annotation:
  - Player bounding boxes with team colors
  - Shot events on court map
  - Broadcast overlay with statistics
- Video output: Annotated frames + court heatmap summary

**Sub-pipelines**:
- `homography_image.py` (658 lines): Single image processing
- `video_pipeline.py` (429 lines): Video batch processing

#### Computer Vision Stack
- **Object Detection**: Ultralytics YOLO v8
- **Inference**: Roboflow inference API
- **Court Transformation**: `sports` package (ViewTransformer, court drawing)
- **Supervision**: Detection/keypoint parsing and manipulation
- **Video I/O**: OpenCV, moviepy, ffmpeg-python
- **Color Processing**: HSV clustering for team classification
- **RANSAC**: Robust homography estimation

#### Configuration Parameters
- Model confidence thresholds: 0.3-0.5
- Homography RANSAC threshold: 5.0 pixels
- Court RMSE quality gate: 1.5 feet (image: 5.0 pixels)
- Keypoint smoothing: 3-frame sliding window
- Event thresholds:
  - Jump shot: 3+ consecutive frames
  - Ball in basket: 2+ consecutive frames
  - Reset timeout: 1.7 seconds
  - Cooldown after made: 0.5 seconds

### 4. React Frontend (`/web`)

#### Components
- **Login.jsx**: JWT authentication form
- **Layout.jsx**: Main application shell with navigation
- **CancerForm.jsx**: Breast cancer prediction input with 30 features
- **IrisForm.jsx**: Iris species prediction with 4 features
- **ModelTraining.jsx**: Async model training UI
- **ResultsDisplay.jsx**: Prediction results visualization
- **MLModelFrontend.jsx**: Model status & readiness dashboard

#### Features
- JWT token management (localStorage)
- Session expiry handling with 401 redirect
- Loading states & error handling
- Real-time model readiness polling
- Form validation (Pydantic schema-aware)
- API error recovery with Retry-After headers

#### Tech Stack
- **Framework**: React 18+ with Vite
- **State**: localStorage for tokens
- **HTTP**: Fetch API with custom retry logic
- **Styling**: CSS modules

---

## Configuration System

### Config Files

**`config.yaml`** - Hierarchical configuration
- Supports: `default`, `dev`, `staging`, `prod` blocks
- Environment variables override YAML values (12-factor app)
- Categories:
  - MLflow: Experiment, tracking/registry URIs, artifact retention
  - Rate Limiting: Separate limits for default/cancer/login/training
  - Quality Gates: Accuracy/F1 thresholds per environment
  - ML Flags: Background training, auto-train missing models
  - Caching: Redis prediction cache with TTL
  - JAX/XLA: Backend device configuration

**Environment Blocks**:
```yaml
dev:  # Relaxed limits, caching enabled, background training
staging: # Stricter limits, self-healing enabled
prod: # Strictest limits, manual approval required
```

**Environment Variables** (highest precedence):
- `ENVIRONMENT` / `APP_ENV`: Selects config block
- `REDIS_URL`: Rate limiting & caching backend
- `DATABASE_URL`: SQLAlchemy connection string
- `MLFLOW_TRACKING_URI`: Model artifact storage
- `CACHE_ENABLED`: Enable/disable prediction caching
- `ROBOFLOW_API_KEY`: CV model API key

---

## Docker & Development Setup

### devcontainer.json Configuration
- **Base Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- **Python**: 3.10-3.12 (3.10 default)
- **GPU**: All NVIDIA GPUs with compute/utility/video capabilities
- **Shared Memory**: 16GB (for data parallelism)
- **Environment Variables**:
  - JAX configuration (GPU memory fraction, XLA flags)
  - PyTorch CUDA settings (memory caching, expanded segments)
  - CV settings (Roboflow API, YOLO verbosity)

### Docker Compose Services
- **datascience**: Main application container
- **mlflow**: Model registry & tracking server (port 5000)
- **jupyter**: Jupyter Lab (port 8888)
- **volumes**: UV cache, YOLO cache, Roboflow cache, MLflow artifacts

### Ports Forwarded
- 8888: Jupyter Lab
- 5000: MLflow UI
- 8050: Explainer Dashboard (SHAP)
- 8501: Streamlit (optional)
- 8080: CV API server
- 8554: RTSP stream (optional)

### Optimization Features
- **Multi-stage build**: Separate cache IDs for PyTorch, JAX, CV packages
- **Build context**: Reduced from 100MB+ to <10MB via enhanced .dockerignore
- **BuildKit**: Parallel builds, caching strategy
- **Validation**: Inline health checks after each major installation

---

## Dependencies & Requirements

### Core Python Stack
```
pandas>=2.0, numpy>=1.26, scipy>=1.7.0, scikit-learn>=1.4.2
matplotlib>=3.4.0, seaborn>=0.11.0, plotly, arviz>=0.14.0
```

### ML/Bayesian
```
pymc>=5.20.0, numpyro>=0.18.0, pytensor>=2.25.0
jax>=0.4.23, jaxlib>=0.4.23
optuna>=4.3.0, mlflow>=3.1.1
xgboost>=1.5.0, lightgbm>=3.3.0, catboost>=1.0.0
```

### Computer Vision
```
ultralytics==8.3.158 (YOLO v8)
opencv-contrib-python-headless>=4.10.0
roboflow==1.2.9 (inference API)
supervision>=0.26,<0.27
inference>=0.61 (GPU inference)
albumentations>=1.3.0, imgaug>=0.4.0, pillow>=10.0.0
sports (GitHub: roboflow/sports - required for court drawing)
```

### Web & API
```
fastapi>=0.100.0, uvicorn>=0.24.0
sqlalchemy>=1.4, aiosqlite, pydantic>=2.0
fastapi-limiter (rate limiting)
redis>=4.5.0 (caching & rate limits)
pydantic-settings>=2.0.0
```

### Video Processing
```
moviepy==2.2.1, pytube>=15.0.0, yt-dlp==2025.9.5
ffmpeg-python==0.2.0 (requires ffmpeg binary)
```

### Development
```
pytest>=7.0.0, black>=23.0.0, isort>=5.0.0
flake8>=5.0.0, mypy>=1.0.0
jupyterlab>=3.0.0, ipywidgets>=8.0.0
```

### PyTorch Configuration
- **CUDA 12.4**: via custom PyTorch index
- **Platform**: Linux/Windows (with markers)
- **Installation**: Via Dockerfile, NOT in pyproject.toml (version negotiation complexity)

---

## Pipeline Stages & Functionality

### Implemented Features

#### 1. Basketball Court Detection
- ✅ Keypoint detection (30 court vertices)
- ✅ Confidence-based filtering
- ✅ Robust RANSAC homography (handles 50%+ outliers)
- ✅ Quality validation (RMSE thresholds)
- ✅ Inverse transform (image ↔ court space)

#### 2. Player Detection & Classification
- ✅ YOLO v8 player bounding boxes
- ✅ Team separation via torso HSV color clustering
- ✅ Referee identification by class/label
- ✅ Anchor point extraction (bottom-center)
- ✅ Coordinate transformation to court space

#### 3. Shot Event Tracking
- ✅ Jump shot detection (3+ frame threshold)
- ✅ Layup/Dunk detection (3+ frame threshold)
- ✅ Ball-in-basket detection (2+ frame threshold)
- ✅ Made/Missed determination
- ✅ Cooldown & reset period handling

#### 4. Visualization & Annotation
- ✅ Player bounding boxes with team colors
- ✅ Court keypoint drawing
- ✅ Shot event markers on court overlay
- ✅ Broadcast annotation (frame counter, shot timer)
- ✅ Court heatmap summary (frame-by-frame player locations)

#### 5. ML Model Serving
- ✅ Iris classification (RF, LogReg)
- ✅ Cancer diagnosis (Bayesian, RF, LogReg)
- ✅ Prediction caching (Redis)
- ✅ Async model training (background tasks)
- ✅ MLflow integration (versioning, promotion)

#### 6. Quality Gates & MLOps
- ✅ Accuracy/F1 threshold checking
- ✅ Staging promotion workflow
- ✅ Production approval flow
- ✅ Model comparison (version A vs B)
- ✅ Hot-reload without restart

---

## Data Processing & Examples

### Notebooks

**Backend Development**:
- `piotr_automated_pipeline.ipynb` - CV pipeline end-to-end
- `backend_api_postworking_fixed_clitested_mladjustments.ipynb` - API testing
- `data_engineering.ipynb` - Data processing & feature engineering

**GPU-Specific**:
- `docker_uv_env_setup_4090_jaxtorchgpu_organized_yolo_updated2_claudecode.ipynb` - RTX 4090 setup
- `yolo_court_detection_codes.ipynb` - YOLO court detection
- `basketball_demo.ipynb` - End-to-end basketball analysis demo
- `environment_uv_docker_jax_pytorch_mlflow_portfriendly_5080_allenabled_efficient_jaxgpu_efficient.ipynb` - RTX 5080 setup

### Sample Data
- Input: Video file (Boston Celtics vs NY Knicks game clip)
- Output:
  - Annotated video with player boxes & shot events
  - JSON analytics (frame-by-frame detections)
  - Court heatmap (shot locations on court)
  - Quality report (detection confidence, homography RMSE)

---

## Testing & Validation

### devcontainer Tests
- `test_pytorch.py` - PyTorch import & version
- `test_yolo.py` - YOLO detection functionality
- `test_cv.py` - OpenCV & supervision
- `test_uv.py` - UV package manager
- `test_pytorch_gpu.py` - GPU availability & CUDA
- `test_summary.py` - Comprehensive validation

### Scripts
- `validate_gpu.py` - GPU detection & capability check
- `verify_env.py` - Environment validation
- `enhanced_gpu_test_functions.py` - GPU-specific tests

---

## Performance & Optimization

### Docker Build Optimization
- **Build Time**: 10-15 minutes (first), 2-5 minutes (cached)
- **Context Transfer**: <5 seconds (vs 470s before optimization)
- **Layer Export**: <60 seconds (vs 325s before)
- **Context Size**: <10MB (vs 100MB+ before)
- **Cache Strategy**: Separate cache IDs per package group (prevents corruption)

### Prediction Caching
- **Backend**: Redis (TTL-based)
- **Keys**: Serialized features + model type + posterior samples
- **Hit Rate**: Depends on workload (testing: 5-30 minute TTL)
- **Overhead**: Minimal serialization/deserialization

### Rate Limiting
- **Default**: 60 requests/minute (120 in dev)
- **Cancer Prediction**: 30 requests/minute (60 in dev)
- **Login**: 3 attempts/20 seconds
- **Training**: 2 jobs/minute
- **Backend**: Redis token bucket

---

## Documentation & Issues

### Known Considerations
1. **sports package**: GitHub dependency (postCreateCommand handles installation)
2. **PyTorch installation**: Handled by Dockerfile (complexity with CUDA variants)
3. **Roboflow API**: Requires API key in environment
4. **Windows builds**: WSL2 recommended (native Windows slow I/O on /mnt/c)

### Recent Optimizations
- ✅ Docker build optimization (2-3x speedup)
- ✅ Enhanced .dockerignore (94x context transfer speedup)
- ✅ Separate cache IDs (fixed tar corruption)
- ✅ Configuration system (centralized YAML + env override)
- ✅ Rate limiting (Redis-backed)
- ✅ Prediction caching (Redis)

---

## Summary Statistics

| Category | Metric | Count/Value |
|----------|--------|------------|
| **Python Modules** | Total files | 23 core + utilities |
| **ML Pipeline** | Stages | 6 (detect, classify, track, evaluate, serve, register) |
| **API Routes** | Endpoints | 40+ (health, auth, predict, train, MLOps, debug) |
| **Models** | Supported types | 2 (Iris, Cancer) × 3 architectures each = 6 total |
| **CV Pipeline** | Lines of code | 5,624 lines |
| **Database** | ORM models | 1 (User) + dynamic |
| **Frontend** | React components | 7 core components |
| **Notebooks** | Development | 14 notebooks (backend, CV, GPU-specific) |
| **Configuration** | Environments | 3 (dev, staging, prod) + default |
| **Dependencies** | Python packages | 60+ (core + optional) |
| **Docker** | Services | 2 (datascience, mlflow) |
| **Ports Exposed** | Count | 7 (Jupyter, MLflow, API, Streamlit, etc.) |

---

## Getting Started Commands

```bash
# Install dependencies
uv sync

# Set up database
npm run seed

# Start backend
npm run backend:dev

# Start frontend (separate terminal)
cd web && npm run dev

# Run tests
npm run test:api

# Check configuration
curl http://localhost:8000/api/v1/debug/effective-config

# View MLflow UI
open http://localhost:5000
```

---

## Deployment Checklist

- [ ] Environment variables configured (REDIS_URL, ROBOFLOW_API_KEY)
- [ ] Redis service running & accessible
- [ ] MLflow volume/directory created
- [ ] Roboflow API key validated
- [ ] GPU available (CUDA 12.4+)
- [ ] Python 3.10-3.12 selected
- [ ] All models trained & in MLflow registry
- [ ] Rate limits configured per environment
- [ ] Frontend API URL updated (VITE_API_URL)
- [ ] Database initialized with user credentials

