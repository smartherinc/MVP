# nnU-Net 3D segmentation web app

Upload a NIfTI in the browser → the server runs **nnU-Net** → the segmentation is
meshed (Taubin-smoothed, 3× x-interpolated, label 1 shown 20% opaque) → it's
rendered as an interactive 3D model with X/Y/Z scale sliders. No more putting
files in folders — just upload and view.

```
backend/    FastAPI: /segment runs nnU-Net + meshing, returns mesh JSON
frontend/   index.html — upload UI + Three.js viewer
data/       nnU-Net working dirs (drop your trained model in nnUNet_results/)
```

## 1. One-time: add your trained model
Copy your trained model into
`data/nnUNet_results/Dataset110_Fibnet_Filtered_nnUnet/nnUNetTrainer__nnUNetPlans__3d_fullres/`:
- `plans.json` and `dataset.json` (in that config folder)
- `fold_0/checkpoint_best.pth` (your checkpoint)

(If your dataset/config/checkpoint names differ, edit them in `.env`.)

## 2. Run the backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or cu121 for GPU
pip install -r requirements-nnunet.txt
cp .env.example .env                                  # adjust if needed

export nnUNet_raw=$(pwd)/../data/nnUNet_raw
export nnUNet_preprocessed=$(pwd)/../data/nnUNet_preprocessed
export nnUNet_results=$(pwd)/../data/nnUNet_results
MODE=nnunet uvicorn app:app --port 8000
```

## 3. Run the frontend
```bash
cd frontend
python -m http.server 5173        # open http://localhost:5173
```
(If your backend isn't on `localhost:8000`, edit `BACKEND_URL` at the top of
`frontend/index.html`.)

Then in the browser: choose your `.nii.gz`, optionally choose a `.pth`
checkpoint, click **Run segmentation**. You'll get the interactive 3D result.

## Test without a model first (optional)
Set `MODE=threshold` to skip nnU-Net and mesh a simple isosurface of the volume,
just to confirm the upload → view loop works end-to-end.

## Tuning the visualization
All in `backend/meshing.py`:
```python
OPACITY = {1: 0.2}                 # per-label transparency
SMOOTH_ITERS = 15                  # Taubin smoothing
INTERP_AXIS = 0; INTERP_FACTOR = 3 # add 3x slices along x
LABEL_NAMES = {1:"Uterus", 2:"Fibroid"}
```

## Deploy on AWS
- **Frontend** → S3 + CloudFront (static). Set `BACKEND_URL` to the API URL and
  the backend's `ALLOW_ORIGINS` to your site origin.
- **Backend** → container (`Dockerfile`) on EC2 `g4dn.*` (GPU, fast nnU-Net) or
  CPU (ECS Fargate). Mount/copy `data/nnUNet_results` into the container.
- nnU-Net inference is slow, especially on CPU — raise your proxy's request
  timeout and upload size limit. For heavy use, switch to an async job + polling
  pattern instead of one long request.
