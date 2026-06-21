# nnU-Net 3D Medical Image Segmentation Web App

A full-stack web application for uploading medical images (NIfTI format) and running **nnU-Net** 3D segmentation inference in real-time. Segmentation results are meshed, smoothed, and rendered interactively in 3D using **Three.js**.

**Key features:** No file system management → upload directly in browser → instant 3D visualization with interactive controls.

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
- GPU (optional, but recommended for inference speed)

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

# PyTorch (choose one):
# For CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

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

**Backend (Docker):**
- Build: `docker build -t nnunet-api .`
- Push to **ECR**
- Deploy on **EC2 g4dn.* (GPU)** or **ECS Fargate (CPU)**
- Mount/COPY `data/nnUNet_results` into container
- Set CORS: `ALLOW_ORIGINS=["https://your-cloudfront-url"]`

**Performance Notes:**
- CPU inference: ~2–10 min/volume (depending on size)
- GPU (g4dn.xlarge): ~10–30 sec/volume
- Increase API request timeout to **300+ seconds**
- Use **async job polling** for production (don't wait 5 min for response)

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

### Slow inference on CPU

This is expected. For production:
- Use GPU (recommended)
- Consider model quantization or pruning
- Implement async job queue (Celery + Redis)

---

## File Format Support

**Input:**
- `.nii` (uncompressed NIfTI)
- `.nii.gz` (gzipped NIfTI) ✅ **Recommended**

**Output (Internal):**
- Mesh JSON (vertices + faces)
- Rendered via Three.js in browser

---

## Tech Stack

| Layer       | Technology          |
|-------------|---------------------|
| Backend     | FastAPI, Python 3.10+|
| ML Engine   | nnU-Net             |
| Processing  | SimpleITK, NumPy    |
| Meshing     | scikit-image, trimesh |
| Smoothing   | Taubin filter       |
| Frontend    | Vanilla JS, Three.js|
| Server      | Uvicorn (ASGI)     |
| Hosting     | AWS EC2/ECS/CloudFront |

---

## Performance Benchmarks

**Tested on AWS g4dn.xlarge (1 GPU):**

| Volume Size | Preprocessing | nnU-Net Inference | Meshing | Total   |
|-------------|---------------|-------------------|---------|---------|
| 256×256×64 | 2 sec         | 8 sec             | 3 sec   | 13 sec  |
| 512×512×128| 5 sec         | 25 sec            | 8 sec   | 38 sec  |
| 512×512×256| 10 sec        | 60+ sec           | 15 sec  | 85+ sec |

**CPU (t3.xlarge):** ~8–10x slower. Use only for testing.

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
