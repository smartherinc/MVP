"""
FastAPI service for the nnU-Net 3D segmentation web app.

Endpoints:
  POST /segment          upload NIfTI -> run nnU-Net -> save result -> return patient id
  GET  /patients         list all patients (metadata only, no heavy mesh data)
  GET  /patients/{id}    return full mesh/volume data for one patient
  DELETE /patients/{id}  delete a patient

Patient data is stored as JSON files in PATIENTS_DIR (default ./patients/).
Only lightweight metadata (name, filename, labels, date) is returned by GET /patients;
the heavy mesh arrays are only returned by GET /patients/{id}, fetched on demand.
This sidesteps the localStorage quota completely.
"""
import os
import json
import uuid
import shutil
import tempfile
import base64
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom, gaussian_filter1d
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from meshing import segmentation_to_meshes

MODE         = os.environ.get("MODE", "nnunet")
MAX_BYTES    = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
PATIENTS_DIR = os.environ.get("PATIENTS_DIR", os.path.join(os.path.dirname(__file__), "patients"))
os.makedirs(PATIENTS_DIR, exist_ok=True)


# ── patient store helpers ────────────────────────────────────────────────────
def _patient_path(pid): return os.path.join(PATIENTS_DIR, f"{pid}.json")

def _load_patient(pid):
    p = _patient_path(pid)
    if not os.path.isfile(p): raise HTTPException(404, f"Patient {pid} not found.")
    with open(p) as f: return json.load(f)

def _save_patient(data):
    with open(_patient_path(data["id"]), "w") as f:
        json.dump(data, f)

def _all_patients():
    out = []
    for fn in os.listdir(PATIENTS_DIR):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(PATIENTS_DIR, fn)) as f:
                d = json.load(f)
            # return only lightweight metadata
            out.append({k: d[k] for k in ("id","name","filename","addedAt","labels","checkpoint") if k in d})
        except Exception:
            continue
    out.sort(key=lambda p: p.get("addedAt", 0), reverse=True)
    return out


# ── processing helpers ───────────────────────────────────────────────────────
def upsample_segmentation_x(seg, affine, factor=3, axis=0, smooth_mm=2.0):
    seg = np.asarray(seg)
    labels = [int(l) for l in np.unique(seg) if l != 0]
    zf = [1.0, 1.0, 1.0]; zf[axis] = float(factor)
    sigma = float(smooth_mm) * float(factor)
    out, best = None, None
    for lab in labels:
        mask = (seg == lab)
        sdf = distance_transform_edt(mask) - distance_transform_edt(~mask)
        sdf = zoom(sdf, zf, order=3)
        if sigma > 0:
            sdf = gaussian_filter1d(sdf, sigma, axis=axis)
        if out is None:
            out = np.zeros(sdf.shape, dtype=seg.dtype)
            best = np.zeros(sdf.shape)
        take = sdf > best
        out[take] = lab; best[take] = sdf[take]
    if out is None:
        out = zoom(seg, zf, order=0).astype(seg.dtype)
    new_affine = affine.copy()
    new_affine[:, axis] = new_affine[:, axis] / factor
    return out, new_affine


def volume_payload(img):
    vol = np.asanyarray(img.dataobj)
    if vol.ndim == 4: vol = vol[..., 0]
    vol = vol.astype(np.float32)
    factor = 1
    while vol.size > 20_000_000:
        vol = vol[::2, ::2, ::2]; factor *= 2
    finite = vol[np.isfinite(vol)]
    lo = float(np.percentile(finite, 1)) if finite.size else 0.0
    hi = float(np.percentile(finite, 99)) if finite.size else 1.0
    if hi <= lo: hi = lo + 1.0
    u8 = np.clip((vol - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    zooms = [float(z) for z in img.header.get_zooms()[:3]] or [1.0, 1.0, 1.0]
    while len(zooms) < 3: zooms.append(1.0)
    return {
        "volume": base64.b64encode(u8.tobytes(order="C")).decode("ascii"),
        "shape": [int(s) for s in u8.shape],
        "spacing": [z * factor for z in zooms[:3]],
    }


# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="nnU-Net 3D segmentation")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE}


@app.get("/patients")
def list_patients():
    return _all_patients()


@app.get("/patients/{pid}")
def get_patient(pid: str):
    """Return full patient data including meshes and volume (fetched on demand)."""
    return _load_patient(pid)


@app.delete("/patients/{pid}")
def delete_patient(pid: str):
    p = _patient_path(pid)
    if not os.path.isfile(p): raise HTTPException(404, f"Patient {pid} not found.")
    os.unlink(p)
    return {"deleted": pid}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    patient_name: str = Form("Unknown"),
    checkpoint: Optional[UploadFile] = File(None),
):
    name = (file.filename or "").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "Please upload a .nii or .nii.gz file.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "NIfTI file too large.")

    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data); path = tmp.name

    ckpt_path = None
    if checkpoint is not None and checkpoint.filename:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as ck:
            shutil.copyfileobj(checkpoint.file, ck); ckpt_path = ck.name

    seg_path = None; used_ckpt = None; vol_pl = None
    try:
        scan_img = nib.load(path)
        try: vol_pl = volume_payload(scan_img)
        except Exception: vol_pl = None

        if MODE == "nnunet":
            from nnunet_runner import run_nnunet
            seg_path = run_nnunet(path, checkpoint_path=ckpt_path)
            simg = nib.load(seg_path)
            seg = np.asanyarray(simg.dataobj); affine = simg.affine
            used_ckpt = (os.path.basename(checkpoint.filename) if ckpt_path
                         else os.environ.get("NNUNET_CHECKPOINT", "checkpoint_final.pth"))
        else:
            vol = np.asanyarray(scan_img.dataobj).astype(np.float32)
            affine = scan_img.affine
            nz = vol[vol > 0]
            thr = float(np.percentile(nz, 60)) if nz.size else 0.5
            seg = (vol > thr).astype(np.uint8)

        seg, affine = upsample_segmentation_x(seg, affine, factor=3, axis=0, smooth_mm=2.0)
        labels, meshes = segmentation_to_meshes(seg, affine)
        if not meshes:
            raise HTTPException(422, "Segmentation has no foreground.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        for p in (path, ckpt_path, seg_path):
            if p and os.path.exists(p): os.unlink(p)

    import time
    patient = {
        "id": str(uuid.uuid4()),
        "name": patient_name,
        "filename": file.filename,
        "addedAt": int(time.time() * 1000),
        "labels": labels,
        "checkpoint": used_ckpt,
        "meshes": meshes,
    }
    if vol_pl: patient.update(vol_pl)
    _save_patient(patient)

    # return only metadata (no heavy arrays) — frontend fetches those on demand
    return {k: patient[k] for k in ("id","name","filename","addedAt","labels","checkpoint")}
