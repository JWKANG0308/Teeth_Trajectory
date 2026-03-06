import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree

from .utils import _safe_int_key
from .registration import slerp_rotation


@dataclass
class LightColCfg:
    enabled: bool = True
    clearance: float = 0.002
    sample_size: int = 200
    neighbor_radius: float = 4.0
    max_delta_p: float = 0.08
    strength: float = 0.7
    seed: int = 0


def build_light_samples(pre_tooth_pts: dict, tooth_ids, sample_size: int, seed: int):
    rng = np.random.default_rng(seed)
    out = {}
    for tid in tooth_ids:
        P = np.asarray(pre_tooth_pts.get(tid, []), dtype=np.float64)
        if P.ndim != 2 or P.shape[0] == 0:
            continue
        if P.shape[0] > sample_size:
            idx = rng.choice(P.shape[0], sample_size, replace=False)
            P = P[idx]
        out[tid] = P
    return out


def build_neighbor_pairs_from_samples(samples: dict, neighbor_radius: float):
    tids = sorted(list(samples.keys()), key=_safe_int_key)
    trees = {tid: cKDTree(samples[tid]) for tid in tids}
    pairs = []
    for i in range(len(tids)):
        a = tids[i]
        Pa = samples[a]
        for j in range(i + 1, len(tids)):
            b = tids[j]
            d, _ = trees[b].query(Pa, k=1)
            if float(d.min()) <= neighbor_radius:
                pairs.append((a, b))
    return pairs


def light_collision_damp_p(
    p_in: dict,
    tooth_transforms: dict,
    samples: dict,
    neighbor_pairs: list,
    clearance: float,
    max_delta_p: float,
    strength: float,
):
    """
    pseudo_intermediate_multi_traj_collision2.py 로직 유지 :contentReference[oaicite:5]{index=5}
    1-pass, gentle damping.
    """
    if clearance <= 0 or max_delta_p <= 0 or strength <= 0:
        return dict(p_in), {"num_pairs_violated": 0, "min_dist": None}

    tids = [
        tid
        for tid in p_in.keys()
        if tid in tooth_transforms and tid in samples and samples[tid].shape[0] > 0
    ]
    if len(tids) == 0 or len(neighbor_pairs) == 0:
        return dict(p_in), {"num_pairs_violated": 0, "min_dist": None}

    p = {tid: float(np.clip(p_in[tid], 0.0, 1.0)) for tid in tids}

    pts = {}
    trees = {}
    for tid in tids:
        P0 = samples[tid]
        R_full, t_full = tooth_transforms[tid]
        R_p = slerp_rotation(R_full, p[tid])
        t_p = p[tid] * t_full
        P = (R_p @ P0.T).T + t_p
        pts[tid] = P
        trees[tid] = cKDTree(P)

    dp = {tid: 0.0 for tid in tids}
    min_dist = np.inf
    violated = 0

    for (a, b) in neighbor_pairs:
        if a not in pts or b not in pts:
            continue
        Pa, Pb = pts[a], pts[b]
        if Pa.shape[0] <= Pb.shape[0]:
            d, _ = trees[b].query(Pa, k=1)
        else:
            d, _ = trees[a].query(Pb, k=1)
        dmin = float(d.min())
        min_dist = min(min_dist, dmin)

        if dmin < clearance:
            violated += 1
            frac = (clearance - dmin) / max(clearance, 1e-9)  # 0..1
            delta = strength * max_delta_p * frac
            dp[a] = max(dp[a], 0.5 * delta)
            dp[b] = max(dp[b], 0.5 * delta)

    for tid in tids:
        if dp[tid] > 0:
            p[tid] = float(np.clip(p[tid] - dp[tid], 0.0, 1.0))

    out = dict(p_in)
    for tid in tids:
        out[tid] = p[tid]

    info = {
        "num_pairs_violated": int(violated),
        "min_dist": (float(min_dist) if np.isfinite(min_dist) else None),
    }
    return out, info
