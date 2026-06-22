"""
FastAPI service for the nnU-Net 3D segmentation web app.

Supported upload types (POST /segment):
  .nii / .nii.gz   NIfTI  → run nnU-Net → mesh the segmentation
  .zip             DICOM  → unzip, sort DICOMs, convert to NIfTI → same nnU-Net flow
  .gltf / .glb     glTF   → skip segmentation, parse meshes directly and store

Endpoints:
  POST /segment          upload file → process → save → return metadata
  GET  /patients         list all patients (metadata only, no heavy mesh data)
  GET  /patients/{id}    return full mesh/volume data for one patient
  DELETE /patients/{id}  delete a patient
"""
import os, json, uuid, shutil, tempfile, base64, zipfile, time
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.ndimage import distance_transform_edt, zoom, gaussian_filter1d

from meshing import segmentation_to_meshes

MODE         = os.environ.get("MODE", "nnunet")
MAX_BYTES    = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
PATIENTS_DIR = os.environ.get("PATIENTS_DIR", os.path.join(os.path.dirname(__file__), "patients"))
os.makedirs(PATIENTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT STORE
# ══════════════════════════════════════════════════════════════════════════════
def _patient_path(pid): return os.path.join(PATIENTS_DIR, f"{pid}.json")

def _load_patient(pid):
    p = _patient_path(pid)
    if not os.path.isfile(p): raise HTTPException(404, f"Patient {pid} not found.")
    with open(p) as f: return json.load(f)

def _save_patient(data):
    with open(_patient_path(data["id"]), "w") as f: json.dump(data, f)

def _all_patients():
    out = []
    for fn in os.listdir(PATIENTS_DIR):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(PATIENTS_DIR, fn)) as f: d = json.load(f)
            out.append({k: d[k] for k in
                ("id","name","filename","fileType","addedAt","labels","checkpoint") if k in d})
        except Exception: continue
    out.sort(key=lambda p: p.get("addedAt", 0), reverse=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FILE TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_file_type(filename: str) -> str:
    n = filename.lower()
    if n.endswith(".nii.gz") or n.endswith(".nii"):  return "nifti"
    if n.endswith(".zip"):                            return "dicom"
    if n.endswith(".gltf") or n.endswith(".glb"):    return "gltf"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# NIFTI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
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
        if sigma > 0: sdf = gaussian_filter1d(sdf, sigma, axis=axis)
        if out is None:
            out = np.zeros(sdf.shape, dtype=seg.dtype); best = np.zeros(sdf.shape)
        take = sdf > best; out[take] = lab; best[take] = sdf[take]
    if out is None: out = zoom(seg, zf, order=0).astype(seg.dtype)
    new_affine = affine.copy(); new_affine[:, axis] = new_affine[:, axis] / factor
    return out, new_affine


def resample_to_isotropic(img, target_spacing=1.0):
    """Resample a NIfTI image to isotropic voxel spacing (default 1x1x1 mm).
    Uses scipy zoom with order=1 (linear) for the scan volume and returns a
    new NIfTI image with an updated affine and identity-like spacing."""
    import nibabel as nib
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    affine = img.affine.copy()

    # current voxel sizes = column norms of the affine
    current_spacing = np.array([
        np.linalg.norm(affine[:3, i]) for i in range(3)
    ])

    zoom_factors = current_spacing / target_spacing
    if np.allclose(zoom_factors, 1.0, atol=0.01):
        return img   # already isotropic, nothing to do

    new_vol = zoom(vol, zoom_factors, order=1).astype(np.float32)

    # update affine: each column direction stays the same, but the step size
    # becomes target_spacing
    new_affine = affine.copy()
    for i in range(3):
        col = affine[:3, i]
        norm = np.linalg.norm(col)
        if norm > 1e-6:
            new_affine[:3, i] = col / norm * target_spacing

    new_img = nib.Nifti1Image(new_vol, new_affine)
    new_img.header.set_qform(new_affine, code=1)
    new_img.header.set_sform(new_affine, code=1)
    return new_img


def volume_payload(img, seg_raw=None, mesh_colors=None):
    """Encode scan as uint8 for the slice viewer.
    If seg_raw (label map on the SAME grid as the scan) and mesh_colors
    ({label_int: [r,g,b] 0-1}) are provided, also encode the segmentation
    so the frontend can overlay colored masks on the slices."""
    import nibabel as nib
    vol = np.asanyarray(img.dataobj)
    if vol.ndim == 4: vol = vol[..., 0]
    vol = vol.astype(np.float32)

    seg_ds = np.asarray(seg_raw).astype(np.uint8) if seg_raw is not None else None

    factor = 1
    while vol.size > 20_000_000:
        vol = vol[::2, ::2, ::2]
        if seg_ds is not None and seg_ds.shape == vol.shape:
            seg_ds = seg_ds[::2, ::2, ::2]
        elif seg_ds is not None:
            seg_ds = None   # shape mismatch after downsample — drop overlay
        factor *= 2

    finite = vol[np.isfinite(vol)]
    lo = float(np.percentile(finite, 1)) if finite.size else 0.0
    hi = float(np.percentile(finite, 99)) if finite.size else 1.0
    if hi <= lo: hi = lo + 1.0
    u8 = np.clip((vol - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    zooms = [float(z) for z in img.header.get_zooms()[:3]] or [1.0, 1.0, 1.0]
    while len(zooms) < 3: zooms.append(1.0)

    out = {
        "volume":  base64.b64encode(u8.tobytes(order="C")).decode("ascii"),
        "shape":   [int(s) for s in u8.shape],
        "spacing": [z * factor for z in zooms[:3]],
    }

    # include segmentation overlay if available and same shape
    if seg_ds is not None and seg_ds.shape == u8.shape and mesh_colors:
        out["segVolume"]  = base64.b64encode(
            np.ascontiguousarray(seg_ds).tobytes(order="C")).decode("ascii")
        # convert {label: [r,g,b]} to list of [label, r255, g255, b255]
        out["segColorMap"] = [
            [int(lab)] + [int(round(c * 255)) for c in rgb]
            for lab, rgb in mesh_colors.items() if lab != 0
        ]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DICOM → NIFTI CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
def dicom_zip_to_nifti(zip_path: str, out_dir: str) -> str:
    """Unzip a DICOM zip, sort slices along their true acquisition direction,
    and convert to NIfTI preserving the original orientation (sagittal stays
    sagittal, axial stays axial, etc.).

    Sorting is done by projecting each slice's ImagePositionPatient onto the
    slice-normal vector (cross product of the row/col direction cosines), which
    is robust for all orientations including sagittal. The affine is built from
    the actual slice positions so spacing is exact.
    """
    try:
        import pydicom
    except ImportError:
        raise HTTPException(500, "pydicom not installed — cannot process DICOM files. "
                                 "Add 'pydicom' to requirements.txt and rebuild.")

    dcm_dir = os.path.join(out_dir, "dcm")
    os.makedirs(dcm_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dcm_dir)

    # ── collect all DICOM files recursively ──────────────────────────────────
    dcm_files = []
    for root, _, files in os.walk(dcm_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                if not hasattr(ds, "SOPClassUID"):
                    continue
                dcm_files.append((fp, ds))
            except Exception:
                pass

    if not dcm_files:
        raise HTTPException(422, "No readable DICOM files found in the zip.")

    # ── get orientation from first slice that has it ──────────────────────────
    iop = None
    for _, ds in dcm_files:
        raw = getattr(ds, "ImageOrientationPatient", None)
        if raw and len(raw) == 6:
            iop = [float(x) for x in raw]
            break

    if iop:
        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:])
        normal  = np.cross(row_cos, col_cos)
        normal  = normal / (np.linalg.norm(normal) + 1e-12)

        # sort by projection of ImagePositionPatient onto the slice normal
        def slice_pos(item):
            ipp = getattr(item[1], "ImagePositionPatient", None)
            if ipp and len(ipp) == 3:
                return float(np.dot([float(x) for x in ipp], normal))
            # fallback: InstanceNumber
            return float(getattr(item[1], "InstanceNumber", 0))

        dcm_files.sort(key=slice_pos)
    else:
        # no orientation info — fall back to InstanceNumber sort
        dcm_files.sort(key=lambda x: (
            int(getattr(x[1], "InstanceNumber", 0)),
            os.path.basename(x[0])
        ))

    # ── read pixel data ───────────────────────────────────────────────────────
    slices = []
    for fp, _ in dcm_files:
        ds = pydicom.dcmread(fp)
        if not hasattr(ds, "pixel_array"):
            continue
        arr = ds.pixel_array.astype(np.float32)
        slope    = float(getattr(ds, "RescaleSlope",     1) or 1)
        intercept= float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        slices.append(ds)

    if not slices:
        raise HTTPException(422, "DICOM files contain no pixel data.")

    # stack slices along axis 0: output shape = (n_slices, rows, cols)
    pixel_data = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope    = float(getattr(ds, "RescaleSlope",     1) or 1)
        intercept= float(getattr(ds, "RescaleIntercept", 0) or 0)
        pixel_data.append(arr * slope + intercept)

    vol = np.stack(pixel_data, axis=0)   # (n_slices, rows, cols)

    # ── compute actual inter-slice spacing from consecutive positions ──────────
    slice_spacing = 1.0   # default fallback
    if iop and len(slices) > 1:
        try:
            row_cos = np.array(iop[:3])
            col_cos = np.array(iop[3:])
            normal  = np.cross(row_cos, col_cos)
            normal  = normal / (np.linalg.norm(normal) + 1e-12)
            ipp0 = np.array([float(x) for x in slices[0].ImagePositionPatient])
            ipp1 = np.array([float(x) for x in slices[1].ImagePositionPatient])
            slice_spacing = abs(float(np.dot(ipp1 - ipp0, normal)))
            if slice_spacing < 0.1:   # implausibly small — fall back
                slice_spacing = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)
        except Exception:
            slice_spacing = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)
    elif len(slices) > 0:
        slice_spacing = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)

    # per-slice transforms (applied to each slice independently):
    #   1. flip horizontally (left-right mirror along columns axis)
    #   2. rotate 90° clockwise  (= rot90 with k=-1, or equivalently k=3)
    # np.rot90(arr, k=1) rotates counter-clockwise; k=-1 (or k=3) is clockwise
    vol = np.flip(vol, axis=2)                     # flip horizontally (cols)
    vol = np.rot90(vol, k=-1, axes=(1, 2)).copy()  # rotate 90° CW in (rows,cols) plane

    import nibabel as nib
    affine = np.eye(4, dtype=np.float64)

    nii = nib.Nifti1Image(vol.astype(np.float64), affine)
    nii.header.set_qform(affine, code=0)
    nii.header.set_sform(affine, code=2)
    nii.header['pixdim'][1:4] = [1.0, 1.0, 1.0]
    out_path = os.path.join(out_dir, "scan.nii.gz")
    nib.save(nii, out_path)
    return out_path, slice_spacing


# ══════════════════════════════════════════════════════════════════════════════
# GLTF PARSER
# ══════════════════════════════════════════════════════════════════════════════
# Default palette for glTF meshes (same as meshing.py)
_PALETTE = [
    [0.90, 0.29, 0.29], [0.29, 0.59, 0.90], [0.37, 0.78, 0.47],
    [0.94, 0.75, 0.27], [0.71, 0.43, 0.86], [0.27, 0.78, 0.78],
    [0.94, 0.51, 0.71], [0.55, 0.80, 0.95], [0.98, 0.62, 0.30],
]

def parse_gltf(file_path: str, filename: str) -> list:
    """Parse a .gltf or .glb file and return a list of mesh dicts compatible
    with the frontend's Three.js renderer (positions, indices, color, opacity)."""
    try:
        import struct

        is_glb = filename.lower().endswith(".glb")

        if is_glb:
            # GLB: binary container — read header + JSON chunk + BIN chunk
            with open(file_path, "rb") as f:
                magic, version, length = struct.unpack("<III", f.read(12))
                if magic != 0x46546C67:  # "glTF"
                    raise ValueError("Not a valid GLB file.")
                # JSON chunk
                j_len, j_type = struct.unpack("<II", f.read(8))
                json_bytes = f.read(j_len)
                gltf = json.loads(json_bytes)
                # BIN chunk (optional)
                bin_data = b""
                remaining = f.read()
                if len(remaining) >= 8:
                    b_len, b_type = struct.unpack("<II", remaining[:8])
                    bin_data = remaining[8:8 + b_len]
        else:
            # glTF: JSON text file; external .bin buffers loaded from same dir
            with open(file_path) as f: gltf = json.load(f)
            bin_data = b""
            base_dir = os.path.dirname(file_path)
            buffers = gltf.get("buffers", [])
            if buffers:
                uri = buffers[0].get("uri", "")
                if uri and not uri.startswith("data:"):
                    bin_path = os.path.join(base_dir, uri)
                    if os.path.isfile(bin_path):
                        with open(bin_path, "rb") as f: bin_data = f.read()
                elif uri.startswith("data:"):
                    # data URI — base64 encoded
                    _, enc = uri.split(",", 1)
                    bin_data = base64.b64decode(enc)

        accessors    = gltf.get("accessors", [])
        buffer_views = gltf.get("bufferViews", [])
        meshes_raw   = gltf.get("meshes", [])

        # accessor dtype map
        COMP_TYPES = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,
                      5125:np.uint32,5126:np.float32}
        COMP_COUNT = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

        def read_accessor(idx):
            acc = accessors[idx]
            bv  = buffer_views[acc["bufferView"]]
            byte_off  = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            count     = acc["count"]
            comp_type = COMP_TYPES.get(acc["componentType"], np.float32)
            n_comp    = COMP_COUNT.get(acc["type"], 1)
            n_bytes   = np.dtype(comp_type).itemsize * n_comp * count
            raw = bin_data[byte_off: byte_off + n_bytes]
            arr = np.frombuffer(raw, dtype=comp_type)
            if n_comp > 1: arr = arr.reshape(count, n_comp)
            return arr

        out_meshes = []
        for mi, mesh in enumerate(meshes_raw):
            mesh_name = mesh.get("name", f"Mesh {mi+1}")
            for pi, prim in enumerate(mesh.get("primitives", [])):
                attrs = prim.get("attributes", {})
                if "POSITION" not in attrs: continue
                pos = read_accessor(attrs["POSITION"]).astype(np.float32)

                if "indices" in prim:
                    idx = read_accessor(prim["indices"]).astype(np.int64).ravel()
                else:
                    # no index buffer — generate sequential indices
                    idx = np.arange(len(pos), dtype=np.int64)

                # color: try to read from material base-color, else pick from palette
                color = _PALETTE[mi % len(_PALETTE)]
                mat_idx = prim.get("material")
                if mat_idx is not None:
                    mats = gltf.get("materials", [])
                    if mat_idx < len(mats):
                        pbr = mats[mat_idx].get("pbrMetallicRoughness", {})
                        bc  = pbr.get("baseColorFactor")
                        if bc and len(bc) >= 3:
                            color = [float(bc[0]), float(bc[1]), float(bc[2])]

                name = mesh_name if len(mesh.get("primitives",[])) == 1 else f"{mesh_name} {pi+1}"
                out_meshes.append({
                    "label":    f"gltf.{mi}.{pi}",
                    "name":     name,
                    "color":    color,
                    "opacity":  1.0,
                    "positions": np.round(pos, 3).ravel().tolist(),
                    "indices":   idx.ravel().tolist(),
                })

        if not out_meshes:
            raise HTTPException(422, "No renderable mesh primitives found in the glTF file.")
        return out_meshes

    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"glTF parsing failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FIXED BODY MODEL (STL) — shown semi-transparent around every patient
# ══════════════════════════════════════════════════════════════════════════════
# Path to the body STL. Override with env BODY_STL_PATH; defaults to body.stl
# sitting next to this file (backend/body.stl).
BODY_STL_PATH = os.environ.get(
    "BODY_STL_PATH", os.path.join(os.path.dirname(__file__), "body.stl"))
BODY_OPACITY  = float(os.environ.get("BODY_OPACITY", "0.10"))   # 10%
BODY_COLOR    = [0.80, 0.80, 0.84]                              # neutral grey

def parse_stl(file_path: str):
    """Parse a binary or ASCII STL into (positions[flat], indices[flat]).
    Deduplicates vertices so the mesh renders with proper shared normals."""
    with open(file_path, "rb") as f:
        head = f.read(5)
    is_ascii = head[:5].lower() == b"solid"
    # an ASCII header can also appear on some binary files; verify by size
    tris = []
    if is_ascii:
        try:
            with open(file_path, "r", errors="ignore") as f:
                cur = []
                for line in f:
                    s = line.strip().split()
                    if len(s) >= 4 and s[0] == "vertex":
                        cur.append([float(s[1]), float(s[2]), float(s[3])])
                        if len(cur) == 3:
                            tris.append(cur); cur = []
            if not tris:
                is_ascii = False           # fall through to binary
        except Exception:
            is_ascii = False

    if not is_ascii:
        import struct
        with open(file_path, "rb") as f:
            f.read(80)                                  # header
            (n,) = struct.unpack("<I", f.read(4))       # triangle count
            for _ in range(n):
                f.read(12)                              # normal (skip)
                v = struct.unpack("<9f", f.read(36))    # 3 vertices
                f.read(2)                               # attribute byte count
                tris.append([[v[0], v[1], v[2]],
                             [v[3], v[4], v[5]],
                             [v[6], v[7], v[8]]])

    if not tris:
        raise ValueError("STL contains no triangles.")

    # dedupe vertices
    verts = {}
    positions = []
    indices = []
    for tri in tris:
        for vx in tri:
            key = (round(vx[0], 4), round(vx[1], 4), round(vx[2], 4))
            idx = verts.get(key)
            if idx is None:
                idx = len(positions) // 3
                verts[key] = idx
                positions.extend([key[0], key[1], key[2]])
            indices.append(idx)
    return positions, indices


_BODY_MESH_CACHE = None   # parse the STL once, reuse for every patient

def get_body_mesh():
    """Return the fixed body as a mesh dict (or None if no STL present)."""
    global _BODY_MESH_CACHE
    if _BODY_MESH_CACHE is not None:
        return _BODY_MESH_CACHE if _BODY_MESH_CACHE else None
    if not os.path.isfile(BODY_STL_PATH):
        _BODY_MESH_CACHE = {}             # mark "checked, none"
        return None
    try:
        positions, indices = parse_stl(BODY_STL_PATH)
        _BODY_MESH_CACHE = {
            "label":   "body",
            "name":    "Body",
            "color":   BODY_COLOR,
            "opacity": BODY_OPACITY,       # 10%
            "positions": positions,
            "indices":   indices,
        }
        return _BODY_MESH_CACHE
    except Exception:
        _BODY_MESH_CACHE = {}
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="nnU-Net 3D segmentation")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://www.mvp.smarthermri.com",
        "https://mvp.smarthermri.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(): return {"status": "ok", "mode": MODE}

@app.get("/patients")
def list_patients(): return _all_patients()

@app.get("/patients/{pid}")
def get_patient(pid: str):
    """Return full patient data with the fixed body model fitted around the anatomy."""
    p = _load_patient(pid)
    if p.get("meshes") and not p.get("fileType") == "gltf":
        p = dict(p)
        rotated = _rotate_anatomy_180(p["meshes"])
        body = get_body_mesh()
        if body:
            p["meshes"] = [_fit_body_to(body, rotated)] + rotated
            p["labels"] = ["body"] + list(p.get("labels") or [])
        else:
            p["meshes"] = rotated
    return p


def _flip_anatomy_vertical(meshes):
    """Flip all anatomy meshes bottom-to-top (negate Y) about their shared
    center, as a rigid unit so uterus and fibroids keep their relative position.
    Triangle winding is reversed so surface normals stay outward."""
    VAX = 2                               # vertical axis (Z)
    allpos = []
    for m in meshes:
        allpos.extend(m["positions"])
    a = np.asarray(allpos, dtype=np.float32).reshape(-1, 3)
    cv = (float(a[:, VAX].min()) + float(a[:, VAX].max())) / 2.0
    out = []
    for m in meshes:
        p = np.asarray(m["positions"], dtype=np.float32).reshape(-1, 3).copy()
        p[:, VAX] = 2.0 * cv - p[:, VAX]
        idx = np.asarray(m["indices"], dtype=np.int64).reshape(-1, 3)
        idx = idx[:, [0, 2, 1]]
        nm = dict(m)
        nm["positions"] = np.round(p, 3).ravel().tolist()
        nm["indices"]   = idx.ravel().tolist()
        out.append(nm)
    return out


def _rotate_anatomy_180(meshes):
    """Mirror all anatomy meshes back-to-front (negate Y, the front-back axis)
    about their shared Y center. X and Z are unchanged.
    Triangle winding is reversed so surface normals stay outward."""
    allpos = []
    for m in meshes:
        allpos.extend(m["positions"])
    a = np.asarray(allpos, dtype=np.float32).reshape(-1, 3)
    cy = (float(a[:, 1].min()) + float(a[:, 1].max())) / 2.0
    out = []
    for m in meshes:
        p = np.asarray(m["positions"], dtype=np.float32).reshape(-1, 3).copy()
        p[:, 1] = 2.0 * cy - p[:, 1]         # mirror Y about center
        idx = np.asarray(m["indices"], dtype=np.int64).reshape(-1, 3)
        idx = idx[:, [0, 2, 1]]               # reverse winding -> normals OK
        nm = dict(m)
        nm["positions"] = np.round(p, 3).ravel().tolist()
        nm["indices"]   = idx.ravel().tolist()
        out.append(nm)
    return out


def _bbox(positions):
    a = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    return a.min(0), a.max(0)

def _fit_body_to(body, anatomy_meshes, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Position the body STL so its center aligns with the center of the
    segmentation (all anatomy meshes combined). Shrinks the body 3x along x.
    offset_x/y/z nudge the body as a fraction of the body's size on that axis."""

    # segmentation center (all anatomy meshes combined)
    allpos = []
    for m in anatomy_meshes:
        allpos.extend(m["positions"])
    amin, amax = _bbox(allpos)
    seg_cen = (amin + amax) / 2.0          # center of the full segmentation

    # body center and vertices
    bpos = np.asarray(body["positions"], dtype=np.float32).reshape(-1, 3)
    bmin, bmax = bpos.min(0), bpos.max(0)
    bcen = (bmin + bmax) / 2.0

    # center body at origin, scale x by 2, then move to segmentation center
    # (seg_cen centers the body on the uterus on all axes including x)
    out = bpos - bcen                      # body centered at origin
    out[:, 0] /= 1                       # scale 2x along x
    out += seg_cen                         # body center = segmentation center

    # optional fine-tuning offsets (fraction of body size per axis)
    bsizes = out.max(0) - out.min(0)
    out[:, 0] += offset_x * bsizes[0]
    out[:, 1] += offset_y * bsizes[1]
    out[:, 2] += offset_z * bsizes[2]

    return {
        "label":   "body",
        "name":    "Body",
        "color":   body["color"],
        "opacity": body["opacity"],
        "positions": np.round(out, 3).ravel().tolist(),
        "indices":   body["indices"],
    }

@app.delete("/patients/{pid}")
def delete_patient(pid: str):
    p = _patient_path(pid)
    if not os.path.isfile(p): raise HTTPException(404, f"Patient {pid} not found.")
    os.unlink(p); return {"deleted": pid}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    patient_name: str = Form("Unknown"),
    checkpoint: Optional[UploadFile] = File(None),
):
    fname    = file.filename or "upload"
    ftype    = detect_file_type(fname)
    if ftype == "unknown":
        raise HTTPException(400,
            "Unsupported file type. Upload a .nii/.nii.gz (NIfTI), "
            ".zip (DICOM), .gltf, or .glb file.")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large.")

    # write to a temp file we can pass to processing functions
    suffix = {
        "nifti": ".nii.gz" if fname.lower().endswith(".nii.gz") else ".nii",
        "dicom": ".zip",
        "gltf":  ".glb" if fname.lower().endswith(".glb") else ".gltf",
    }[ftype]
    tmp_dir = tempfile.mkdtemp()
    upload_path = os.path.join(tmp_dir, f"upload{suffix}")
    with open(upload_path, "wb") as f: f.write(raw)

    ckpt_path = None
    if checkpoint is not None and checkpoint.filename:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as ck:
            shutil.copyfileobj(checkpoint.file, ck); ckpt_path = ck.name

    patient = {
        "id":       str(uuid.uuid4()),
        "name":     patient_name,
        "filename": fname,
        "fileType": ftype,
        "addedAt":  int(time.time() * 1000),
    }

    try:
        # ── glTF: no segmentation, just parse and store meshes directly ──────
        if ftype == "gltf":
            meshes = parse_gltf(upload_path, fname)
            patient.update({
                "labels":     [m["label"] for m in meshes],
                "checkpoint": None,
                "meshes":     meshes,
            })

        # ── NIfTI or DICOM: run nnU-Net ──────────────────────────────────────
        else:
            import nibabel as nib

            if ftype == "dicom":
                nifti_path, slice_spacing = dicom_zip_to_nifti(upload_path, tmp_dir)
            else:
                nifti_path   = upload_path
                slice_spacing = 3.0   # NIfTI default (original factor)

            scan_img = nib.load(nifti_path)
            vol_pl = None

            seg_path  = None
            used_ckpt = None
            overlay_scan   = None   # per-fibroid ID volume on the scan grid
            overlay_colors = None   # {int_id: [r,g,b]}
            try:
                if MODE == "nnunet":
                    from nnunet_runner import run_nnunet
                    seg_path = run_nnunet(nifti_path, checkpoint_path=ckpt_path)
                    simg     = nib.load(seg_path)
                    seg      = np.asanyarray(simg.dataobj)
                    affine   = simg.affine
                    used_ckpt = (os.path.basename(checkpoint.filename) if ckpt_path
                                 else os.environ.get("NNUNET_CHECKPOINT","checkpoint_final.pth"))
                else:
                    vol    = np.asanyarray(scan_img.dataobj).astype(np.float32)
                    affine = scan_img.affine
                    nz     = vol[vol > 0]
                    thr    = float(np.percentile(nz, 60)) if nz.size else 0.5
                    seg    = (vol > thr).astype(np.uint8)

                scan_shape = np.asanyarray(scan_img.dataobj).shape[:3]

                # upsample segmentation along the slice axis by the actual inter-slice
                # spacing in mm so the mesh has isotropic resolution:
                #   NIfTI: factor=3 (original hardcoded default for thick-slice data)
                #   DICOM: factor = real slice spacing in mm (e.g. 5.5mm -> factor=5.5)
                upsample_factor = round(float(slice_spacing), 2)
                seg_up, affine_up = upsample_segmentation_x(
                    seg, affine, factor=upsample_factor, axis=0, smooth_mm=2.0)
                # split happens here; overlay_vol holds a unique ID per fibroid
                labels, meshes, overlay_vol, overlay_colors = segmentation_to_meshes(seg_up, affine_up)
                if not meshes:
                    raise HTTPException(422, "Segmentation has no foreground.")

                # back-project per-fibroid IDs to scan grid
                if overlay_vol.shape == scan_shape:
                    overlay_scan = overlay_vol.astype(np.uint8)
                else:
                    try:
                        factors = [s / t for s, t in zip(scan_shape, overlay_vol.shape)]
                        overlay_scan = zoom(overlay_vol.astype(np.float32), factors,
                                            order=0).astype(np.uint8)
                    except Exception:
                        overlay_scan = None

            except HTTPException: raise
            except Exception as e: raise HTTPException(500, f"Processing failed: {e}")
            finally:
                if seg_path and os.path.exists(seg_path): os.unlink(seg_path)
                if ckpt_path and os.path.exists(ckpt_path): os.unlink(ckpt_path)

            # per-fibroid overlay: each fibroid its own color, uterus grey base
            try:
                vol_pl = volume_payload(scan_img, seg_raw=overlay_scan,
                                        mesh_colors=overlay_colors)
            except Exception:
                vol_pl = None

            patient.update({
                "labels":     labels,
                "checkpoint": used_ckpt,
                "meshes":     meshes,
            })
            if vol_pl: patient.update(vol_pl)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_patient(patient)
    return {k: patient[k] for k in
            ("id","name","filename","fileType","addedAt","labels","checkpoint") if k in patient}
