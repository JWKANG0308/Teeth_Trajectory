import numpy as np
from scipy.spatial import cKDTree
from .utils import _safe_int_key


def build_tooth_vertex_indices(mesh_vertices: np.ndarray, tooth_points: dict, tol: float):
    tree = cKDTree(mesh_vertices)
    owner = -np.ones(len(mesh_vertices), dtype=np.int32)
    tooth_indices = {}

    for tooth_id in sorted(tooth_points.keys(), key=_safe_int_key):
        pts = np.asarray(tooth_points[tooth_id], dtype=np.float64)
        if pts.size == 0:
            tooth_indices[tooth_id] = np.array([], dtype=np.int64)
            continue

        d, idx = tree.query(pts, k=1)
        idx = idx[d < tol]
        if idx.size == 0:
            tooth_indices[tooth_id] = np.array([], dtype=np.int64)
            continue

        idx = np.unique(idx)
        free = owner[idx] < 0
        idx = idx[free]

        try:
            owner_id = int(tooth_id)
        except Exception:
            owner_id = 1
        owner[idx] = owner_id

        tooth_indices[tooth_id] = idx.astype(np.int64)

    return tooth_indices


def build_tooth_face_masks(faces: np.ndarray, tooth_indices: dict):
    face_masks = {}
    for tid, vidx in tooth_indices.items():
        if vidx is None or len(vidx) == 0:
            continue
        in_tooth = np.isin(faces, vidx)  # (F,3)
        mask = np.all(in_tooth, axis=1)  # (F,)
        if np.any(mask):
            face_masks[tid] = mask
    return face_masks
