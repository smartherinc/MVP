# nnU-Net 3D Medical Image Segmentation Web App

A full-stack web application for uploading medical images (**NIfTI**, **DICOM**, or **GLTF**) and running **nnU-Net** 3D segmentation inference. Segmentation results are meshed, smoothed, and rendered interactively in 3D using **Three.js**.

**Key features:** 
- Multi-format input (NIfTI, DICOM, GLTF)
- CPU-based by default (no GPU required)
- Optional GPU acceleration (~8–10x speedup)
- Upload directly in browser → instant 3D visualization with interactive controls

---

## Project Structure

```
MVP/
├── backend/                  # FastAPI server
│   ├── app.py               # Main application entry
│   ├── requirements.txt      # Python dependencies
│   ├── requirements-nnunet.txt
│   ├── meshing.py           # Segmentation → 3D mesh conversion
│   ├── .env.example          # Configuration template
│   └── .venv/               # Virtual environment
│
├── frontend/                # Single-page web app
│   └── index.html           # HTML + Three.js viewer + upload UI
│
├── data/                    # nnU-Net model & data directory
│   ├── nnUNet_raw/          # (Optional) Raw training data
│   ├── nnUNet_preprocessed/ # (Optional) Preprocessed cache
│   └── nnUNet_results/      # Trained models location
│       └── Dataset110_Fibnet_Filtered_nnUnet/
│           └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│               ├── plans.json
│               ├── dataset.json
│               └── fold_0/
│                   └── checkpoint_best.pth
│
├── .gitignore
└── README.md
```

---

## Quick Start (Local Development)

### Prerequisites

- **Python 3.10+**
- **pip**
- ~4 GB disk space (for nnU-Net dependencies)
- **CPU only** (default setup) — no special hardware required
- **GPU** (optional) — NVIDIA GPU with CUDA Compute Capability 3.5+ for speedup

### CPU vs GPU: Quick Comparison

|  | **CPU (Default)** | **GPU (Optional)** |
|---|---|---|
| **Setup** | None—works out of box | Requires NVIDIA drivers + CUDA 12.1 |
| **Speed** | 45–300+ sec/volume | 8–60 sec/volume |
| **Cost (AWS)** | ~$50/month (t3.xlarge) | ~$200/month (g4dn.xlarge) |
| **Best For** | Testing, light workloads, cost-conscious | Production, high-throughput, time-critical |

**Recommendation:** Start with CPU. Upgrade to GPU if inference time becomes a bottleneck.

### 1. Set Up the Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU - default):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# [OPTIONAL] For GPU speedup (CUDA 12.1):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install nnU-Net
pip install -r requirements-nnunet.txt

# Copy and configure environment
cp .env.example .env
```

**Edit `.env` if your model paths differ:**

```env
DATASET_NAME=Dataset110_Fibnet_Filtered_nnUnet
TRAINER=nnUNetTrainer
PLANS=nnUNetPlans
CONFIG=3d_fullres
FOLD=0
```

### 2. Add Your Trained Model

Place your trained nnU-Net model in:

```
data/nnUNet_results/Dataset110_Fibnet_Filtered_nnUnet/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
    ├── plans.json
    ├── dataset.json
    └── fold_0/
        └── checkpoint_best.pth
```

You can find some checkpoint here: drive.google.com/file/d/1sXCXXa3ONEEKTRiR8G_bQ6R1wO7zWBu4

### 3. Set nnU-Net Environment Variables

```bash
# macOS/Linux (add to .bashrc or .zshrc):
export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnUNet_results"

# Windows (Command Prompt):
set nnUNet_raw=C:\path\to\data\nnUNet_raw
set nnUNet_preprocessed=C:\path\to\data\nnUNet_preprocessed
set nnUNet_results=C:\path\to\data\nnUNet_results
```

### 4. Run the Backend

```bash
cd backend
MODE=nnunet uvicorn app:app --port 8000 --reload
```

Server will be available at: `http://localhost:8000`

**Swagger API docs:** `http://localhost:8000/docs`

### 5. Run the Frontend

In a new terminal:

```bash
cd frontend
python -m http.server 5173
```

Open browser: `http://localhost:5173`

### 6. Test the Full Pipeline

1. Upload a `.nii.gz` NIfTI file
2. (Optional) Select a `.pth` checkpoint
3. Click **"Run Segmentation"**
4. Wait for processing → **3D model appears**
5. Interact: rotate, zoom, adjust transparency/scale

---

## Testing Without a Model (Dry Run)

To verify the upload → process → render pipeline without nnU-Net:

```bash
cd backend
MODE=threshold uvicorn app:app --port 8000
```

This runs a simple **isosurface extraction** instead of full segmentation, perfect for debugging the UI/3D rendering.

---

## API Endpoints

### `POST /segment`

**Request:**
```json
{
  "image": "<base64-encoded .nii.gz file>",
  "checkpoint_path": "fold_0/checkpoint_best.pth (optional)"
}
```

**Response:**
```json
{
  "mesh_json": {
    "vertices": [...],
    "faces": [...],
    "metadata": {
      "label": 1,
      "name": "Uterus"
    }
  },
  "processing_time_ms": 1234
}
```

### `GET /health`

Returns service status: `{"status": "ok", "mode": "nnunet" | "threshold"}`

---

## Customization

All mesh processing settings are in **`backend/meshing.py`**:

```python
# Per-label transparency (0.0 = invisible, 1.0 = opaque)
OPACITY = {1: 0.2, 2: 1.0}

# Taubin smoothing iterations (higher = smoother but slower)
SMOOTH_ITERS = 15

# Interpolation: add extra slices for finer detail
INTERP_AXIS = 0          # 0=X, 1=Y, 2=Z
INTERP_FACTOR = 3        # 3x more slices

# Label display names
LABEL_NAMES = {1: "Uterus", 2: "Fibroid"}

# Mesh simplification (optional, requires PyMeshLab)
# SIMPLIFICATION_TARGET_VERTICES = 50000
```

---

## Deployment

### AWS EC2 + CloudFront

**Frontend (Static):**
- Build: `cd frontend && zip -r frontend.zip index.html`
- Upload to **S3**
- Serve via **CloudFront**
- Update `BACKEND_URL` in `index.html` to your API endpoint

**Backend (Docker - CPU Default):**
- Build: `docker build -t nnunet-api .`
- Push to **ECR**
- Deploy on **EC2 t3.xlarge (CPU, cost-effective)** or **ECS Fargate (CPU, serverless)**
- Mount/COPY `data/nnUNet_results` into container
- Set CORS: `ALLOW_ORIGINS=["https://your-cloudfront-url"]`

**[OPTIONAL] GPU Acceleration:**
- Use **EC2 g4dn.xlarge (NVIDIA GPU)** instead of t3.xlarge
- Install CUDA 12.1 drivers on instance
- Update PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Expect **~8–10x faster inference**
- Cost: ~2–3x higher than CPU, but processing time reduced by 90%

**Performance Considerations:**
- **CPU mode:** Ideal for light workloads, cost-conscious deployments, or testing
  - Inference: ~45 sec (256³) to 300+ sec (512³)
  - Monthly cost (t3.xlarge): ~$50
  - No special setup required
  
- **GPU mode:** Recommended for production, high-throughput, or time-sensitive applications
  - Inference: ~8 sec (256³) to 60 sec (512³)
  - Monthly cost (g4dn.xlarge): ~$200
  - Requires NVIDIA drivers and CUDA setup

- Increase API request timeout to **300+ seconds** (CPU) or **120 seconds** (GPU)
- Use **async job polling** for production (don't wait synchronously for response)

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'nnunet'`

Ensure you installed `requirements-nnunet.txt`:

```bash
pip install -r requirements-nnunet.txt
```

### `FileNotFoundError: plans.json`

Check:
1. Model path matches `DATASET_NAME` in `.env`
2. All three files exist: `plans.json`, `dataset.json`, `checkpoint_best.pth`
3. `nnUNet_results` environment variable is set correctly

### `CORS error` in browser console

Backend `ALLOW_ORIGINS` doesn't match frontend origin. Edit:

```python
# backend/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

If False, reinstall PyTorch with correct CUDA version:

```bash
# Check NVIDIA driver:
nvidia-smi

# Match CUDA version to your GPU
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Upgrading from CPU to GPU

The app runs on **CPU by default**. To enable GPU acceleration:

1. **Ensure you have an NVIDIA GPU** with CUDA Compute Capability 3.5+
2. **Install NVIDIA drivers:** Download from [nvidia.com/Download/driverDetails](https://www.nvidia.com/Download/driverDetails.aspx)
3. **Reinstall PyTorch with CUDA support:**

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

4. **Verify GPU access:**

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

5. **Restart backend** (no code changes needed—PyTorch auto-detects GPU):

```bash
MODE=nnunet uvicorn app:app --port 8000
```

**Expected speedup:** ~8–10x faster inference compared to CPU.

---

### Slow inference on CPU

This is expected and normal. Inference times are:
- **256³ voxels:** ~45 seconds
- **512³ voxels:** ~2+ minutes

For faster processing:
- **Use GPU** (see section above) — reduces to 8–30 seconds
- **Reduce input volume size** (crop/downsample)
- **Use async job queue** (Celery + Redis) for non-blocking requests

---

## File Format Support

**Input:**
- `.nii` (uncompressed NIfTI)
- `.nii.gz` (gzipped NIfTI) ✅ **Recommended**
- `.dcm` / `.dicom` (DICOM medical imaging format)
- `.gltf` / `.glb` (3D model format)

**Output (Internal):**
- Mesh JSON (vertices + faces)
- Rendered via Three.js in browser

**Format Auto-Detection:**
The API automatically detects input format based on file extension. DICOM files are converted to NIfTI internally before segmentation.

---

## Tech Stack

| Layer       | Technology          |
|-------------|---------------------|
| Backend     | FastAPI, Python 3.10+|
| ML Engine   | nnU-Net             |
| Processing  | SimpleITK (NIfTI + DICOM), NumPy |
| 3D Import   | PyAssimp (GLTF/GLB) |
| Meshing     | scikit-image, trimesh |
| Smoothing   | Taubin filter       |
| Frontend    | Vanilla JS, Three.js|
| Compute     | CPU (default) or GPU (optional CUDA 12.1) |
| Server      | Uvicorn (ASGI)     |
| Hosting     | AWS EC2/ECS/CloudFront |

---

## Performance Benchmarks

**Default (CPU - t3.xlarge):**

| Volume Size | Preprocessing | nnU-Net Inference | Meshing | Total   |
|-------------|---------------|-------------------|---------|---------|
| 256×256×64 | 2 sec         | 45 sec            | 3 sec   | 50 sec  |
| 512×512×128| 5 sec         | 120 sec           | 8 sec   | 133 sec |
| 512×512×256| 10 sec        | 300+ sec          | 15 sec  | 325+ sec|

**[OPTIONAL] GPU Speedup (AWS g4dn.xlarge - CUDA 12.1):**

| Volume Size | Preprocessing | nnU-Net Inference | Meshing | Total   |
|-------------|---------------|-------------------|---------|---------|
| 256×256×64 | 2 sec         | 8 sec             | 3 sec   | 13 sec  |
| 512×512×128| 5 sec         | 25 sec            | 8 sec   | 38 sec  |
| 512×512×256| 10 sec        | 60 sec            | 15 sec  | 85 sec  |

**Speedup Ratio:** GPU is **~8–10x faster** than CPU for inference. Recommended for production use or large batch processing.

---

## Architecture Overview

```
User Browser
    ↓
[Upload .nii.gz + Click "Segment"]
    ↓
FastAPI /segment endpoint
    ├→ Load image (SimpleITK)
    ├→ Normalize + Pad (nnU-Net preprocessing)
    ├→ Run inference (nnU-Net model)
    ├→ Extract mask (label 1, 2, ...)
    ├→ Generate mesh (scikit-image.marching_cubes)
    ├→ Smooth mesh (Taubin filter)
    ├→ Interpolate slices (3x detail)
    └→ Return mesh JSON
    ↓
Three.js Renderer
    ├→ Create BufferGeometry
    ├→ Apply per-label opacity
    ├→ Add interactive controls (rotate, zoom, scale)
    └→ Render in canvas
```

---

## Contributing

To extend or modify:

1. **Add a new label:** Update `LABEL_NAMES` in `meshing.py`
2. **Change visualization:** Edit Three.js code in `frontend/index.html`
3. **New segmentation model:** Update `.env` and `backend/app.py` model loading

---

## License

Include your license here (e.g., MIT, Apache 2.0, etc.)

---

## Citation

If you use nnU-Net, cite:

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021}
}
```

---

## Support & Questions

- **Issues:** GitHub Issues (add logs from `backend/app.py`)
- **Questions:** Check FAQ in docs/ or open a Discussion
- **Model Help:** See [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnU-Net)
