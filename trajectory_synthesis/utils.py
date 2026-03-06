# src/utils.py
import os
import json
import numpy as np


def _safe_int_key(x: str):
    try:
        return int(x)
    except Exception:
        return x


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def load_segmentation_vertices(json_path: str):
    """Read { 'segmentation': { tooth_id: {'vertices': [[x,y,z], ...]}, ... } }"""
    with open(json_path, "r") as f:
        data = json.load(f)
    seg = data.get("segmentation", {}) or {}
    out = {}
    for tooth_id, obj in seg.items():
        if isinstance(obj, dict) and "vertices" in obj:
            pts = np.asarray(obj["vertices"], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                out[str(tooth_id)] = pts
    return out


def is_valid_tooth_id(tid: str):
    """
    evaluation_matrix_pseudo.py 스타일 유지 :contentReference[oaicite:3]{index=3}
    Default: numeric IDs 10..99 and last digit not 0
    """
    try:
        v = int(tid)
        if v == 0:
            return False
        if v < 10 or v > 99:
            return False
        if (v % 10) == 0:
            return False
        return True
    except Exception:
        return False


def sample_points(P: np.ndarray, n: int, rng: np.random.Generator):
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] <= n:
        return P.copy()
    idx = rng.choice(P.shape[0], n, replace=False)
    return P[idx]
