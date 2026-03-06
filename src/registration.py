import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def rigid_transform_kabsch(A: np.ndarray, B: np.ndarray):
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = cB - (Rm @ cA)
    return Rm, t


def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def icp_rigid(
    source: np.ndarray,
    target: np.ndarray,
    max_iter=30,
    tol=1e-6,
    sample_size=1024,
    seed=0,
):
    rng = np.random.default_rng(seed)
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)

    if src.shape[0] > sample_size:
        src = src[rng.choice(src.shape[0], sample_size, replace=False)]
    if tgt.shape[0] > sample_size:
        tgt = tgt[rng.choice(tgt.shape[0], sample_size, replace=False)]

    tree = cKDTree(tgt)

    R_total = np.eye(3)
    t_total = np.zeros(3)

    prev_err = np.inf
    for _ in range(max_iter):
        src_trans = (R_total @ src.T).T + t_total
        d, idx = tree.query(src_trans, k=1)
        matched = tgt[idx]

        R_delta, t_delta = rigid_transform_kabsch(src_trans, matched)
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        mean_err = float(np.mean(d))
        if abs(prev_err - mean_err) < tol:
            break
        prev_err = mean_err

    return R_total, t_total, prev_err


def estimate_tooth_transform(
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    *,
    min_tooth_points: int = 200,
    kabsch_rmse_thresh: float = 0.2,
    icp_max_iter: int = 30,
    icp_tol: float = 1e-6,
    icp_sample_size: int = 1024,
    icp_seed: int = 0,
):

    pre_pts = np.asarray(pre_pts, dtype=np.float64)
    post_pts = np.asarray(post_pts, dtype=np.float64)

    if pre_pts.shape[0] < min_tooth_points or post_pts.shape[0] < min_tooth_points:
        raise ValueError(f"Too few points (pre={pre_pts.shape[0]}, post={post_pts.shape[0]})")

    # Try Kabsch if same size
    if pre_pts.shape[0] == post_pts.shape[0]:
        n = min(pre_pts.shape[0], icp_sample_size)
        rng = np.random.default_rng(icp_seed)
        if pre_pts.shape[0] > n:
            idx = rng.choice(pre_pts.shape[0], n, replace=False)
            A, B = pre_pts[idx], post_pts[idx]
        else:
            A, B = pre_pts, post_pts
        Rk, tk = rigid_transform_kabsch(A, B)
        e = rmse((Rk @ A.T).T + tk, B)
        if e <= kabsch_rmse_thresh:
            return Rk, tk, float(e), "kabsch"

    # Fallback ICP
    Ri, ti, e = icp_rigid(
        pre_pts,
        post_pts,
        max_iter=icp_max_iter,
        tol=icp_tol,
        sample_size=icp_sample_size,
        seed=icp_seed,
    )
    return Ri, ti, float(e), "icp"


def slerp_rotation(R_target: np.ndarray, alpha: float) -> np.ndarray:
    key_rots = Rotation.from_matrix([np.eye(3), R_target])
    slerp = Slerp([0, 1], key_rots)
    return slerp([alpha]).as_matrix()[0]
