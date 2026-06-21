"""
Turn a segmentation label map into renderable mesh data, with Taubin smoothing
and a postprocessing step that splits the fibroid label into individual
fibroids (connected components), each rendered as its own colored mesh.

Returns plain dicts (positions/indices/color/opacity/name/label) that the
browser turns into Three.js BufferGeometry — no trimesh/GLB needed. The
frontend already renders one colored, toggleable mesh per dict, so multiple
fibroids appear as separate toggles with distinct colors automatically.
"""
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import label as cc_label
from scipy.ndimage import (distance_transform_edt, maximum_filter,
                           binary_fill_holes, binary_closing,
                           generate_binary_structure, iterate_structure)
from skimage import measure
from skimage.segmentation import watershed

# ---- visual / processing config (tuned for the Fibroid/Uterus model) --------
PALETTE = [
    [0.90, 0.29, 0.29], [0.29, 0.59, 0.90], [0.37, 0.78, 0.47],
    [0.94, 0.75, 0.27], [0.71, 0.43, 0.86], [0.27, 0.78, 0.78],
    [0.94, 0.51, 0.71], [0.55, 0.80, 0.95], [0.98, 0.62, 0.30],
    [0.60, 0.85, 0.55], [0.85, 0.55, 0.95], [0.63, 0.63, 0.63],
]
UTERUS_COLOR = [0.78, 0.78, 0.82]        # fixed neutral color for the uterus shell

LABEL_NAMES = {1: "Uterus", 2: "Fibroid"}
OPACITY = {1: 0.2}                       # label 1 shown 20% opaque; others solid

SMOOTH_ITERS = 15                        # Taubin iterations (0 = off)
SMOOTH_ITERS_PER_LABEL = {1: 18, 2: 12}

# Postprocessing: separate each label into individual fibroids in two stages:
#   STAGE 1 — connectedness: disconnected blobs are separate fibroids.
#   STAGE 2 — protrusions:   within a connected blob, a lobe that bulges out past
#             a thin NECK is split off as its own fibroid. A bump that is *not*
#             clearly separated (the neck is nearly as thick as the lobes) stays
#             part of the main round body.
SPLIT_LABELS = {2}
MIN_COMPONENT_VOXELS = 50    # drop specks smaller than this (noise)
PEAK_MIN_DISTANCE = 5        # min separation (voxels) between candidate lobe centers
PEAK_REL_HEIGHT = 0.35       # a lobe peak must be >= this * its blob's max radius
NECK_RATIO = 0.75            # split a lobe off only if the neck connecting it is
                             # thinner than this * the smaller lobe's radius.
                             # lower = more conservative (only sharply pinched
                             # lobes split); higher = splits even gentle bulges.
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


# Mask cleanup for the uterus shell so its surface has no holes/gaps.
CLOSE_RADIUS = 2          # morphological-closing radius (voxels) to seal thin gaps

def clean_mask(mask, close_radius=CLOSE_RADIUS):
    """Make a mask watertight for meshing: morphologically close thin gaps and
    pinholes, then fill any interior cavities. Removes the broken/empty patches
    that marching cubes produces on thin or pitted surfaces."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    if close_radius > 0:
        st = iterate_structure(generate_binary_structure(3, 1), int(close_radius))
        m = binary_closing(m, structure=st)
    m = binary_fill_holes(m)          # fill interior cavities → solid, closed shell
    return m


def mesh_from_mask(mask, affine, smooth_iters=0, watertight=False):
    """Marching cubes on a boolean mask -> (verts in world mm, faces).
    watertight=True cleans the mask (close + fill holes) first so the surface
    has no gaps — used for the uterus shell."""
    mask = np.asarray(mask, dtype=bool)
    if watertight:
        mask = clean_mask(mask)
    # pad generously so a mask touching the array edge is still closed off
    m = np.pad(mask.astype(np.float32), 2, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(m, level=0.5)
    verts = verts - 2.0
    verts = (affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    if smooth_iters > 0 and len(faces):
        verts = taubin_smooth(verts, faces, iterations=smooth_iters)
    return verts.astype(np.float32), faces.astype(np.int64)


def _mesh_dict(label, name, color, opacity, v, f):
    return {
        "label": label,
        "name": name,
        "color": color,
        "opacity": opacity,
        "positions": np.round(v, 2).ravel().tolist(),
        "indices": f.ravel().tolist(),
    }


def _split_blob_protrusions(comp, dist, min_distance, rel_height, neck_ratio):
    """Within ONE connected blob, split off lobes that bulge past a thin neck.

    1. find lobe centers = distance-map peaks
    2. watershed → one basin per lobe (cut at the necks/valleys)
    3. merge back any two basins whose neck is nearly as thick as the lobes
       (i.e. not a real protrusion — just one round body)
    Returns a label image over `comp` (1..k).
    """
    conn = np.ones((3, 3, 3), int)

    # 1. candidate lobe centers
    peak = (maximum_filter(dist, size=2 * min_distance + 1) == dist) & comp
    thr = rel_height * dist[comp].max()
    peak &= ~(comp & (dist < thr))
    markers, nm = cc_label(peak, structure=conn)
    if not markers.any():                      # guarantee one seed
        idx = np.argmax(np.where(comp, dist, -1.0))
        markers.flat[idx] = 1; nm = 1
    if nm <= 1:
        return comp.astype(np.int32)           # one round thing, no protrusion

    # 2. watershed → basins cut at the necks
    ws = watershed(-dist, markers, mask=comp)

    # peak (lobe radius) per basin
    nb = int(ws.max())
    peak_val = np.zeros(nb + 1)
    for b in range(1, nb + 1):
        sel = ws == b
        if sel.any(): peak_val[b] = dist[sel].max()

    # 3. neck thickness on every basin-basin interface (the "mountain pass")
    #    neck_val[(a,b)] = max dist along their shared boundary
    neck = {}
    idxs = np.argwhere(ws > 0)
    for (x, y, z) in idxs:
        a = ws[x, y, z]
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            xx, yy, zz = x + dx, y + dy, z + dz
            if xx < ws.shape[0] and yy < ws.shape[1] and zz < ws.shape[2]:
                b = ws[xx, yy, zz]
                if b > 0 and b != a:
                    key = (a, b) if a < b else (b, a)
                    val = max(dist[x, y, z], dist[xx, yy, zz])
                    if val > neck.get(key, -1.0): neck[key] = val

    # union-find: merge basins whose neck is NOT thin enough to be a protrusion
    parent = list(range(nb + 1))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[max(ri, rj)] = min(ri, rj)

    for (a, b), neck_val in neck.items():
        lobe = min(peak_val[a], peak_val[b])
        # neck nearly as thick as the smaller lobe -> same round body -> merge
        if lobe <= 0 or neck_val >= neck_ratio * lobe:
            union(a, b)

    # relabel by merged root
    out = np.zeros_like(ws, dtype=np.int32)
    remap, k = {}, 0
    for b in range(1, nb + 1):
        r = find(b)
        if r not in remap:
            k += 1; remap[r] = k
        out[ws == b] = remap[r]
    return out


def split_components(mask, min_voxels=MIN_COMPONENT_VOXELS,
                     min_distance=PEAK_MIN_DISTANCE, rel_height=PEAK_REL_HEIGHT,
                     neck_ratio=NECK_RATIO):
    """Two-stage fibroid split.

    STAGE 1 (connectedness): label disconnected blobs — each is its own fibroid.
    STAGE 2 (protrusions):   inside each blob, split off lobes that stand out past
                             a thin neck; gentle bumps stay with the main body.
    Returns a label image (1..k), 0 = background.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0

    conn = np.ones((3, 3, 3), int)
    blobs, nb = cc_label(mask, structure=conn)        # STAGE 1

    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 0
    for b in range(1, nb + 1):
        comp = blobs == b
        dist = distance_transform_edt(comp)
        sub = _split_blob_protrusions(comp, dist, min_distance, rel_height, neck_ratio)  # STAGE 2
        for s in range(1, int(sub.max()) + 1):
            sel = sub == s
            if sel.sum() >= min_voxels:
                next_id += 1
                out[sel] = next_id
    return out, next_id


def segmentation_to_meshes(seg, affine):
    """-> (labels, [mesh dicts], overlay_vol, overlay_colors)

    overlay_vol    : uint16 array same shape as seg. Each voxel holds an
                     integer ID that indexes into overlay_colors. 0 = background.
    overlay_colors : dict {int_id: [r,g,b]} in 0-1 floats — one entry per mesh,
                     so every split fibroid gets its own distinct color.
    """
    seg = np.asarray(seg)
    labels = [int(l) for l in np.unique(seg) if l != 0]
    meshes = []
    color_idx = 0

    # overlay volume: same shape as seg, each voxel → unique int ID (1-based)
    overlay_vol    = np.zeros(seg.shape, dtype=np.uint16)
    overlay_colors = {}   # {int_id: [r,g,b]}
    next_id        = 1    # 0 = background

    for lab in labels:
        iters     = SMOOTH_ITERS_PER_LABEL.get(lab, SMOOTH_ITERS)
        opacity   = OPACITY.get(lab, 1.0)
        base_name = LABEL_NAMES.get(lab, f"label {lab}")

        if lab in SPLIT_LABELS:
            comp, n = split_components(seg == lab)
            sizes = np.bincount(comp.ravel())
            comps = [c for c in range(1, n + 1) if sizes[c] >= MIN_COMPONENT_VOXELS]
            comps.sort(key=lambda c: sizes[c], reverse=True)   # largest first
            for k, c in enumerate(comps, start=1):
                try:
                    v, f = mesh_from_mask(comp == c, affine, iters)
                except (ValueError, RuntimeError):
                    color_idx += 1; next_id += 1; continue
                color = PALETTE[color_idx % len(PALETTE)]
                meshes.append(_mesh_dict(
                    f"{lab}.{k}", f"{base_name} {k}", color, opacity, v, f))
                # assign this fibroid's unique ID to its voxels in the overlay
                overlay_vol[comp == c] = next_id
                overlay_colors[next_id] = color
                color_idx += 1
                next_id   += 1
        else:
            try:
                mask = (seg > 0) if lab == 1 else (seg == lab)
                # uterus (label 1): mesh watertight so the shell has no gaps
                v, f = mesh_from_mask(mask, affine, iters, watertight=(lab == 1))
            except (ValueError, RuntimeError):
                continue
            color = UTERUS_COLOR if lab == 1 else PALETTE[color_idx % len(PALETTE)]
            meshes.append(_mesh_dict(lab, base_name, color, opacity, v, f))
            # uterus: assign its own ID only to the uterus voxels (label==1),
            # not the fibroids inside it (those get their own IDs above)
            mask_seg = (seg == lab)
            overlay_vol[mask_seg] = next_id
            overlay_colors[next_id] = color
            next_id += 1
            if lab != 1:
                color_idx += 1

    return labels, meshes, overlay_vol, overlay_colors
