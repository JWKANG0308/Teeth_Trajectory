import os
import json
import csv
import glob
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .utils import _safe_int_key, load_segmentation_vertices, is_valid_tooth_id, sample_points, ensure_dir


@dataclass
class EvalConfig:
    margin: float = 0.0015
    sample_size: int = 500
    pair_mode: str = "neighbors"  # "all" / "neighbors"
    neighbor_radius: float = 4.0
    include_extracted_before_removal: bool = True
    seed: int = 0
    verbose: bool = True


def quat_xyzw_to_R(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64)  # [x,y,z,w]
    return Rotation.from_quat(q).as_matrix()


def build_neighbor_pairs(tooth_samples: dict, neighbor_radius: float):
    tooth_ids = sorted(list(tooth_samples.keys()), key=_safe_int_key)
    trees = {tid: cKDTree(tooth_samples[tid]) for tid in tooth_ids}
    pairs = []
    for a_i in range(len(tooth_ids)):
        a = tooth_ids[a_i]
        Pa = tooth_samples[a]
        for b_i in range(a_i + 1, len(tooth_ids)):
            b = tooth_ids[b_i]
            d, _ = trees[b].query(Pa, k=1)
            if float(d.min()) <= neighbor_radius:
                pairs.append((a, b))
    return pairs


def compute_pair_min_dist(Pa, treeB, Pb, treeA=None):
    if Pa.shape[0] <= Pb.shape[0]:
        d, _ = treeB.query(Pa, k=1)
    else:
        if treeA is None:
            treeA = cKDTree(Pa)
        d, _ = treeA.query(Pb, k=1)
    return float(np.min(d))


def evaluate_multi_traj_collision(out_root: str, ori_json_path: str, cfg: EvalConfig, save_dir: str = None):
    """
    원본 로직 기반 :contentReference[oaicite:6]{index=6}
    out_root:
      case_dir/pseudo_staging_lightcol_{JAW}/traj_XX/poses.json ...
    ori_json_path:
      case_dir/ori/{JAW}_Ori.json
    """
    if save_dir is None:
        save_dir = out_root
    ensure_dir(save_dir)

    pre_pts = load_segmentation_vertices(ori_json_path)
    pre_pts = {tid: P for tid, P in pre_pts.items() if is_valid_tooth_id(tid)}
    if len(pre_pts) < 2:
        raise RuntimeError("Not enough tooth point clouds in ori json to evaluate collisions.")

    rng = np.random.default_rng(cfg.seed)

    tooth_samples_ori = {}
    for tid in sorted(pre_pts.keys(), key=_safe_int_key):
        tooth_samples_ori[tid] = sample_points(pre_pts[tid], cfg.sample_size, rng)

    # pairs
    if cfg.pair_mode == "neighbors":
        pairs = build_neighbor_pairs(tooth_samples_ori, cfg.neighbor_radius)
        if cfg.verbose:
            print(f"[INFO] pair_mode=neighbors | neighbor_pairs={len(pairs)} (radius={cfg.neighbor_radius})")
        if len(pairs) == 0:
            if cfg.verbose:
                print("[WARN] neighbor pairs empty -> fallback to pair_mode=all")
            cfg = EvalConfig(**{**cfg.__dict__, "pair_mode": "all"})
    if cfg.pair_mode == "all":
        tids = sorted(tooth_samples_ori.keys(), key=_safe_int_key)
        pairs = [(tids[i], tids[j]) for i in range(len(tids)) for j in range(i + 1, len(tids))]

    # traj dirs
    traj_dirs = []
    for name in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, name)
        if os.path.isdir(p) and name.startswith("traj_"):
            if os.path.exists(os.path.join(p, "poses.json")):
                traj_dirs.append(p)
    if len(traj_dirs) == 0:
        raise FileNotFoundError(f"No traj_*/poses.json found under: {out_root}")

    summary_rows = []

    for traj_path in traj_dirs:
        traj_name = os.path.basename(traj_path)
        poses_path = os.path.join(traj_path, "poses.json")
        with open(poses_path, "r") as f:
            meta = json.load(f)

        poses = meta.get("poses", [])
        if len(poses) == 0:
            print(f"[WARN] {traj_name}: poses is empty -> skip")
            continue

        extracted_teeth = [str(x) for x in meta.get("extracted_teeth", [])]
        extraction_remove_step = int(meta.get("extraction_remove_step", 10))

        all_teeth = set(tooth_samples_ori.keys())

        tooth_ids_sorted = sorted(list(all_teeth), key=_safe_int_key)
        n = len(tooth_ids_sorted)
        tid_to_idx = {tid: i for i, tid in enumerate(tooth_ids_sorted)}

        energy_mat = np.zeros((n, n), dtype=np.float64)
        count_mat = np.zeros((n, n), dtype=np.int32)
        mindist_mat = np.full((n, n), np.inf, dtype=np.float64)

        per_step_min = []
        per_step_energy = []
        per_step_any_violation = []

        margin = float(cfg.margin)

        if cfg.verbose:
            print(f"\n[INFO] Evaluating {traj_name} | steps={len(poses)} | margin={margin}")

        for step_idx, step_pose in enumerate(poses):
            extracted_removed = bool(step_pose.get("extracted_removed", False))

            active = set(all_teeth)
            if (not cfg.include_extracted_before_removal) or extracted_removed or (step_idx >= extraction_remove_step):
                active = active - set(extracted_teeth)

            active = sorted(list(active), key=_safe_int_key)
            if len(active) < 2:
                per_step_min.append(np.inf)
                per_step_energy.append(0.0)
                per_step_any_violation.append(False)
                continue

            step_teeth = step_pose.get("teeth", {}) or {}

            pts_step = {}
            trees = {}

            for tid in active:
                P0 = tooth_samples_ori.get(tid, None)
                if P0 is None or P0.shape[0] == 0:
                    continue

                if tid in step_teeth:
                    quat = step_teeth[tid]["quat_xyzw"]
                    txyz = step_teeth[tid]["t_xyz"]
                    Rm = quat_xyzw_to_R(quat)
                    t = np.asarray(txyz, dtype=np.float64)
                    P = (Rm @ P0.T).T + t
                else:
                    P = P0

                pts_step[tid] = P
                trees[tid] = cKDTree(P)

            step_min = np.inf
            step_energy = 0.0
            step_any_vio = False

            for (a, b) in pairs:
                if a not in pts_step or b not in pts_step:
                    continue
                if (a not in tid_to_idx) or (b not in tid_to_idx):
                    continue
                ia, ib = tid_to_idx[a], tid_to_idx[b]

                Pa, Pb = pts_step[a], pts_step[b]
                da = compute_pair_min_dist(Pa, trees[b], Pb, treeA=trees[a])

                if da < mindist_mat[ia, ib]:
                    mindist_mat[ia, ib] = da
                    mindist_mat[ib, ia] = da

                step_min = min(step_min, da)

                if da < margin:
                    step_any_vio = True
                    v = (margin - da)
                    e = v * v
                    step_energy += e
                    energy_mat[ia, ib] += e
                    energy_mat[ib, ia] += e
                    count_mat[ia, ib] += 1
                    count_mat[ib, ia] += 1

            per_step_min.append(float(step_min))
            per_step_energy.append(float(step_energy))
            per_step_any_violation.append(bool(step_any_vio))

        per_step_min_arr = np.asarray(per_step_min, dtype=np.float64)
        per_step_energy_arr = np.asarray(per_step_energy, dtype=np.float64)
        vio_arr = np.asarray(per_step_any_violation, dtype=bool)

        finite = per_step_min_arr[np.isfinite(per_step_min_arr)]
        global_min_dist = float(np.min(finite) if finite.size > 0 else np.inf)
        mean_step_min_dist = float(np.mean(finite) if finite.size > 0 else np.inf)

        total_clear_energy = float(np.sum(per_step_energy_arr))
        mean_clear_energy_per_step = float(np.mean(per_step_energy_arr))

        violation_step_ratio = float(np.mean(vio_arr))
        violation_step_count = int(np.sum(vio_arr))

        row = {
            "traj": traj_name,
            "num_steps": len(poses),
            "margin": margin,
            "pair_mode": cfg.pair_mode,
            "neighbor_radius": cfg.neighbor_radius if cfg.pair_mode == "neighbors" else "",
            "include_extracted_before_removal": cfg.include_extracted_before_removal,
            "global_min_dist": global_min_dist,
            "mean_step_min_dist": mean_step_min_dist,
            "total_clear_energy": total_clear_energy,
            "mean_clear_energy_per_step": mean_clear_energy_per_step,
            "violation_step_ratio": violation_step_ratio,
            "violation_step_count": violation_step_count,
        }
        summary_rows.append(row)

        # save per-step curves
        per_step_out = os.path.join(save_dir, f"{traj_name}_per_step_metrics.csv")
        with open(per_step_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "min_dist_step", "clear_energy_step", "any_violation"])
            for i in range(len(poses)):
                w.writerow([i, per_step_min[i], per_step_energy[i], int(per_step_any_violation[i])])

        # matrices (csv with labels)
        def save_matrix_csv(mat, path):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tid"] + tooth_ids_sorted)
                for i, tid in enumerate(tooth_ids_sorted):
                    w.writerow([tid] + list(mat[i]))

        save_matrix_csv(energy_mat, os.path.join(save_dir, f"{traj_name}_pair_energy.csv"))
        save_matrix_csv(count_mat.astype(np.int32), os.path.join(save_dir, f"{traj_name}_pair_count.csv"))

        mindist_out = mindist_mat.copy()
        mindist_out[~np.isfinite(mindist_out)] = -1.0
        save_matrix_csv(mindist_out, os.path.join(save_dir, f"{traj_name}_pair_mindist.csv"))

        if cfg.verbose:
            print(
                f"[DONE] {traj_name}: global_min_dist={global_min_dist:.4f}, "
                f"total_clear_energy={total_clear_energy:.4f}, "
                f"violation_step_ratio={violation_step_ratio:.3f}"
            )

    summary_csv = os.path.join(save_dir, "collision_summary_by_traj.csv")
    if len(summary_rows) > 0:
        keys = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    print(f"\n[ALL DONE] Saved summary -> {summary_csv}")
    return summary_rows, summary_csv


# ---- heatmap (optional) ----
def make_collision_heatmaps(save_dir: str, margin: float):
    import pandas as pd
    import matplotlib.pyplot as plt

    def _read_labeled_matrix_csv(path: str):
        df = pd.read_csv(path)
        labels = df.iloc[:, 0].astype(str).tolist()
        mat = df.iloc[:, 1:].to_numpy()
        return labels, mat

    def _heatmap(mat, labels, title, out_path, mask_value=None, vmin=None, vmax=None, rotate_xticks=90):
        M = mat.astype(float).copy()
        mask = np.zeros_like(M, dtype=bool)
        if mask_value is not None:
            mask |= (M == mask_value)
        mask |= np.eye(M.shape[0], dtype=bool)
        M[mask] = np.nan

        plt.figure(figsize=(10, 8))
        im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(labels)), labels, rotation=rotate_xticks, fontsize=8)
        plt.yticks(np.arange(len(labels)), labels, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    mindist_files = sorted(glob.glob(os.path.join(save_dir, "traj_*_pair_mindist.csv")))
    if len(mindist_files) == 0:
        raise FileNotFoundError(f"No traj_*_pair_mindist.csv found in {save_dir}")

    for md_path in mindist_files:
        base = os.path.basename(md_path).replace("_pair_mindist.csv", "")
        count_path = os.path.join(save_dir, f"{base}_pair_count.csv")
        energy_path = os.path.join(save_dir, f"{base}_pair_energy.csv")

        labels, md = _read_labeled_matrix_csv(md_path)
        _, cnt = _read_labeled_matrix_csv(count_path)
        _, eng = _read_labeled_matrix_csv(energy_path)

        md_png = os.path.join(save_dir, f"{base}_heatmap_mindist.png")
        _heatmap(md, labels, f"{base} | Pair MinDist over time | danger < {margin}", md_png,
                 mask_value=-1.0, vmin=0.0, vmax=max(margin * 2.0, 1e-6))

        cnt_png = os.path.join(save_dir, f"{base}_heatmap_count.png")
        _heatmap(cnt, labels, f"{base} | Violation Step Count (d < {margin})", cnt_png,
                 mask_value=None, vmin=0.0, vmax=np.nanmax(cnt) if np.isfinite(np.nanmax(cnt)) else None)

        eng_png = os.path.join(save_dir, f"{base}_heatmap_energy.png")
        _heatmap(eng, labels, f"{base} | Violation Energy Sum Σ(margin-d)^2", eng_png,
                 mask_value=None, vmin=0.0, vmax=np.nanmax(eng) if np.isfinite(np.nanmax(eng)) else None)

        print(f"[SAVED] {md_png}")
        print(f"[SAVED] {cnt_png}")
        print(f"[SAVED] {eng_png}")

    print("\n[ALL DONE] Heatmaps generated.")


# ---- margin suggestion (optional) ----
@dataclass
class MarginSuggestConfig:
    sample_size: int = 800
    pair_mode: str = "neighbors"  # "neighbors" or "all"
    neighbor_radius: float = 4.0
    seed: int = 0
    max_pairs: int = 5000
    verbose: bool = True


def suggest_margin_from_ori(ori_json_path: str, cfg: MarginSuggestConfig = MarginSuggestConfig(), out_csv_path: str | None = None):
    pre_pts = load_segmentation_vertices(ori_json_path)
    pre_pts = {tid: P for tid, P in pre_pts.items() if is_valid_tooth_id(tid)}
    if len(pre_pts) < 2:
        raise RuntimeError("Not enough tooth point clouds in ori json.")

    rng = np.random.default_rng(cfg.seed)

    tooth_samples = {}
    for tid in sorted(pre_pts.keys(), key=_safe_int_key):
        tooth_samples[tid] = sample_points(pre_pts[tid], cfg.sample_size, rng)

    tids = sorted(tooth_samples.keys(), key=_safe_int_key)
    if cfg.pair_mode == "neighbors":
        pairs = build_neighbor_pairs(tooth_samples, cfg.neighbor_radius)
        if cfg.verbose:
            print(f"[INFO] pair_mode=neighbors | neighbor_pairs={len(pairs)} (radius={cfg.neighbor_radius})")
        if len(pairs) == 0:
            if cfg.verbose:
                print("[WARN] neighbor pairs empty -> fallback to pair_mode=all")
            cfg = MarginSuggestConfig(**{**cfg.__dict__, "pair_mode": "all"})

    if cfg.pair_mode == "all":
        pairs = [(tids[i], tids[j]) for i in range(len(tids)) for j in range(i + 1, len(tids))]
        if len(pairs) > cfg.max_pairs:
            idx = rng.choice(len(pairs), cfg.max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]
        if cfg.verbose:
            print(f"[INFO] pair_mode=all | pairs_used={len(pairs)}")

    trees = {tid: cKDTree(tooth_samples[tid]) for tid in tids}
    pair_dists = []
    rows = []
    for (a, b) in pairs:
        da = compute_pair_min_dist(tooth_samples[a], trees[b], tooth_samples[b], treeA=trees[a])
        pair_dists.append(da)
        rows.append((a, b, da))

    pair_dists = np.asarray(pair_dists, dtype=np.float64)
    pair_dists = pair_dists[np.isfinite(pair_dists)]
    if pair_dists.size == 0:
        raise RuntimeError("No finite distances computed.")

    qs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    qvals = {q: float(np.quantile(pair_dists, q)) for q in qs}

    suggestions = {
        "margin_contactish_q01": qvals[0.01],
        "margin_safer_q05": qvals[0.05],
        "margin_very_safe_q10": qvals[0.10],
        "common_candidates": [0.00, 0.02, 0.05, 0.08, 0.10, 0.25],
    }

    stats = {
        "num_teeth": len(tids),
        "pair_mode": cfg.pair_mode,
        "neighbor_radius": cfg.neighbor_radius if cfg.pair_mode == "neighbors" else None,
        "pairs_used": int(len(pairs)),
        "dist_min": float(pair_dists.min()),
        "dist_mean": float(pair_dists.mean()),
        "dist_median": float(np.median(pair_dists)),
        "dist_max": float(pair_dists.max()),
        "quantiles": qvals,
        "suggestions": suggestions,
    }

    if out_csv_path is not None:
        ensure_dir(os.path.dirname(out_csv_path) or ".")
        with open(out_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tooth_a", "tooth_b", "ori_min_dist"])
            for a, b, da in rows:
                w.writerow([a, b, da])
        if cfg.verbose:
            print(f"[SAVED] ORI pair distances -> {out_csv_path}")

    if cfg.verbose:
        print("\n==== ORI MinDist Distribution (pairs) ====")
        print(f"min    : {stats['dist_min']:.6f}")
        print(f"median : {stats['dist_median']:.6f}")
        print(f"mean   : {stats['dist_mean']:.6f}")
        print(f"max    : {stats['dist_max']:.6f}")
        print("---- suggested margins ----")
        print(f"contact-ish (q1%) : {suggestions['margin_contactish_q01']:.6f}")
        print(f"safer      (q5%) : {suggestions['margin_safer_q05']:.6f}")
        print(f"very safe (q10%) : {suggestions['margin_very_safe_q10']:.6f}")
        print("common candidates:", suggestions["common_candidates"])

    return stats


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser("Collision evaluation for pseudo trajectories")
    p.add_argument("--out_root", type=str, required=True, help=".../pseudo_staging_lightcol_{JAW}")
    p.add_argument("--ori_json", type=str, required=True, help=".../ori/{JAW}_Ori.json")
    p.add_argument("--margin", type=float, default=0.0015)
    p.add_argument("--sample_size", type=int, default=500)
    p.add_argument("--pair_mode", type=str, default="neighbors", choices=["neighbors", "all"])
    p.add_argument("--neighbor_radius", type=float, default=4.0)
    p.add_argument("--include_extracted_before_removal", action="store_true")
    p.add_argument("--exclude_extracted_before_removal", dest="include_extracted_before_removal", action="store_false")
    p.set_defaults(include_extracted_before_removal=True)

    p.add_argument("--save_heatmaps", action="store_true")
    p.add_argument("--heatmap_margin", type=float, default=None, help="if None, uses --margin")

    p.add_argument("--suggest_margin", action="store_true")
    p.add_argument("--suggest_out_csv", type=str, default=None)
    return p


def main():
    args = _build_argparser().parse_args()

    cfg = EvalConfig(
        margin=args.margin,
        sample_size=args.sample_size,
        pair_mode=args.pair_mode,
        neighbor_radius=args.neighbor_radius,
        include_extracted_before_removal=args.include_extracted_before_removal,
    )
    evaluate_multi_traj_collision(args.out_root, args.ori_json, cfg)

    if args.save_heatmaps:
        hm_margin = args.heatmap_margin if args.heatmap_margin is not None else args.margin
        make_collision_heatmaps(args.out_root, margin=float(hm_margin))

    if args.suggest_margin:
        suggest_margin_from_ori(args.ori_json, out_csv_path=args.suggest_out_csv)


if __name__ == "__main__":
    main()
