"""
Turn a segmentation label map into renderable mesh data (one mesh per label),
with Taubin smoothing and shape-based interpolation along a coarse axis.

Returns plain dicts (positions/indices/color/opacity) that the browser turns
into Three.js BufferGeometry — no trimesh/GLB needed.
"""
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt, zoom
from skimage import measure

# ---- visual / processing config (tuned for the Fibroid/Uterus model) --------
PALETTE = [
    [0.90, 0.29, 0.29], [0.29, 0.59, 0.90], [0.37, 0.78, 0.47],
    [0.94, 0.75, 0.27], [0.71, 0.43, 0.86], [0.27, 0.78, 0.78],
    [0.94, 0.51, 0.71], [0.63, 0.63, 0.63],
]
LABEL_NAMES = {1: "Uterus", 2: "Fibroid"}
OPACITY = {1: 0.2}                       # label 1 shown 20% opaque; others solid

SMOOTH_ITERS = 4                        # Taubin iterations (0 = off)
SMOOTH_ITERS_PER_LABEL = {1: 2, 2: 4}

INTERP_AXIS = 0                          # axis to upsample (0 = first array axis = world x for identity affine)
INTERP_FACTOR = 1                        # x-interpolation now done as postprocessing in app.py (see upsample_segmentation_x)
INTERP_FACTOR_PER_LABEL = {}             # optional per-label overrides
# -----------------------------------------------------------------------------


def taubin_smooth(verts, faces, iterations=15, lamb=0.5, mu=-0.53):
    """Round the surface with minimal shrinkage (lambda/mu smoothing)."""
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    avg = sp.diags(1.0 / deg) @ A
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v += lamb * (avg @ v - v)
        v += mu * (avg @ v - v)
    return v


def mesh_label(seg, affine, lab, smooth_iters=0, interp_factor=1, interp_axis=0):
    mask = (seg == lab)
    if interp_factor and interp_factor > 1:
        sdf = distance_transform_edt(mask) - distance_transform_edt(~mask)
        zf = [1.0, 1.0, 1.0]
        zf[interp_axis] = float(interp_factor)
        sdf = zoom(sdf, zf, order=1)
        sdf = np.pad(sdf, 1, mode="constant", constant_values=float(sdf.min()))
        verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
        verts = verts - 1.0
        verts[:, interp_axis] /= interp_factor
    else:
        m = np.pad(mask.astype(np.float32), 1, mode="constant")
        verts, faces, _, _ = measure.marching_cubes(m, level=0.5)
        verts = verts - 1.0
    verts = (affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    if smooth_iters > 0 and len(faces):
        verts = taubin_smooth(verts, faces, iterations=smooth_iters)
    return verts.astype(np.float32), faces.astype(np.int64)


def segmentation_to_meshes(seg, affine):
    """-> (labels, [mesh dicts]). Empty list if no foreground."""
    labels = [int(l) for l in np.unique(seg) if l != 0]
    meshes = []
    for idx, lab in enumerate(labels):
        iters = SMOOTH_ITERS_PER_LABEL.get(lab, SMOOTH_ITERS)
        factor = INTERP_FACTOR_PER_LABEL.get(lab, INTERP_FACTOR)
        try:
            v, f = mesh_label(seg, affine, lab, iters, factor, INTERP_AXIS)
        except (ValueError, RuntimeError):
            continue
        meshes.append({
            "label": lab,
            "name": LABEL_NAMES.get(lab, f"label {lab}"),
            "color": PALETTE[idx % len(PALETTE)],
            "opacity": OPACITY.get(lab, 1.0),
            "positions": np.round(v, 2).ravel().tolist(),
            "indices": f.ravel().tolist(),
        })
    return labels, meshes
