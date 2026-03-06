import os
import json
import numpy as np
import trimesh
from dataclasses import asdict
from scipy.spatial.transform import Rotation

from .utils import _safe_int_key, load_segmentation_vertices, ensure_dir
from .registration import estimate_tooth_transform, slerp_rotation
from .mesh_ops import build_tooth_vertex_indices, build_tooth_face_masks
from .schedule import ScheduleConfig, sample_schedule_params, progress_with_delay_and_warp
from .collision_light import LightColCfg, build_light_samples, build_neighbor_pairs_from_samples, light_collision_damp_p


def generate_multi_pseudo_staging_with_extraction(
    case_dir: str,
    jaw: str,
    *,
    num_steps: int = 20,
    num_traj: int = 8,
    out_dir: str | None = None,
    export_stl: bool = True,
    save_poses_json: bool = True,
    extraction_remove_alpha: float = 0.10,
    vertex_match_tol: float = 1e-4,
    # registration params
    min_tooth_points: int = 200,
    kabsch_rmse_thresh: float = 0.2,
    icp_max_iter: int = 30,
    icp_tol: float = 1e-6,
    icp_sample_size: int = 1024,
    icp_seed: int = 0,
    # schedule + collision cfg
    sched_cfg: ScheduleConfig = ScheduleConfig(),
    light_cfg: LightColCfg = LightColCfg(),
):
    jaw = jaw.upper()
    ori_stl = os.path.join(case_dir, "ori", f"{jaw}_Ori.stl")
    ori_json = os.path.join(case_dir, "ori", f"{jaw}_Ori.json")
    fin_json = os.path.join(case_dir, "final", f"{jaw}_Final.json")

    if not (os.path.exists(ori_stl) and os.path.exists(ori_json)):
        raise FileNotFoundError(f"Missing ori files: {ori_stl} / {ori_json}")
    if not os.path.exists(fin_json):
        raise FileNotFoundError(f"Missing final json: {fin_json}")

    out_root = out_dir or os.path.join(case_dir, f"pseudo_staging_lightcol_{jaw}")
    ensure_dir(out_root)

    pre_mesh = trimesh.load_mesh(ori_stl, process=True)
    pre_mesh.remove_unreferenced_vertices()

    pre_tooth_pts = load_segmentation_vertices(ori_json)
    post_tooth_pts = load_segmentation_vertices(fin_json)

    pre_teeth = set(pre_tooth_pts.keys())
    post_teeth = set(post_tooth_pts.keys())

    extracted_teeth = sorted(list(pre_teeth - post_teeth), key=_safe_int_key)
    appearing_teeth = sorted(list(post_teeth - pre_teeth), key=_safe_int_key)
    common_teeth = sorted(list(pre_teeth & post_teeth), key=_safe_int_key)

    print(f"[INFO] Jaw={jaw}")
    print(f"  common teeth: {len(common_teeth)}")
    print(f"  extracted (pre-only): {extracted_teeth}")
    print(f"  appearing (post-only): {appearing_teeth}")

    # 1) estimate transforms for common teeth
    tooth_transforms = {}
    tooth_report = []
    for tooth_id in common_teeth:
        try:
            Rm, t, err, method = estimate_tooth_transform(
                pre_tooth_pts[tooth_id],
                post_tooth_pts[tooth_id],
                min_tooth_points=min_tooth_points,
                kabsch_rmse_thresh=kabsch_rmse_thresh,
                icp_max_iter=icp_max_iter,
                icp_tol=icp_tol,
                icp_sample_size=icp_sample_size,
                icp_seed=icp_seed,
            )
            tooth_transforms[tooth_id] = (Rm, t)
            tooth_report.append({"tooth_id": tooth_id, "method": method, "fit_error": err})
        except Exception as e:
            tooth_report.append({"tooth_id": tooth_id, "method": "skip", "reason": str(e)})

    used_teeth = sorted(list(tooth_transforms.keys()), key=_safe_int_key)
    print(f"[INFO] usable teeth (transform estimated) = {len(used_teeth)}")

    # indices per tooth on Ori mesh
    tooth_indices = build_tooth_vertex_indices(pre_mesh.vertices, pre_tooth_pts, tol=vertex_match_tol)

    # face masks for extraction removal
    face_masks = build_tooth_face_masks(pre_mesh.faces, tooth_indices)
    extracted_face_mask = None
    for tid in extracted_teeth:
        m = face_masks.get(tid, None)
        if m is None:
            continue
        extracted_face_mask = m if extracted_face_mask is None else (extracted_face_mask | m)

    remove_step = int(np.round(extraction_remove_alpha * (num_steps - 1)))
    remove_step = int(np.clip(remove_step, 0, num_steps - 1))
    print(f"[INFO] extracted teeth will be removed from step >= {remove_step} (alpha~{extraction_remove_alpha})")

    # LIGHT collision
    light_samples = build_light_samples(pre_tooth_pts, used_teeth, light_cfg.sample_size, light_cfg.seed)
    light_pairs = build_neighbor_pairs_from_samples(light_samples, light_cfg.neighbor_radius)
    print(f"[INFO] light collision: samples={len(light_samples)} teeth, neighbor_pairs={len(light_pairs)}")

    base_rng = np.random.default_rng(sched_cfg.seed)

    for k in range(num_traj):
        traj_dir = os.path.join(out_root, f"traj_{k:02d}")
        ensure_dir(traj_dir)

        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        start_delay, gamma = sample_schedule_params(used_teeth, sched_cfg, rng)

        poses = []
        for step in range(num_steps):
            s_global = step / (num_steps - 1) if num_steps > 1 else 1.0
            verts = pre_mesh.vertices.copy()

            step_pose = {
                "s_global": float(s_global),
                "extracted_removed": bool(step >= remove_step),
                "light_collision": {},
                "teeth": {},
            }

            # compute p for all teeth first
            p_dict = {tid: progress_with_delay_and_warp(s_global, start_delay[tid], gamma[tid]) for tid in used_teeth}

            # LIGHT collision damping (skip last step to preserve final)
            if light_cfg.enabled and step != (num_steps - 1) and len(light_pairs) > 0:
                p_dict, light_info = light_collision_damp_p(
                    p_in=p_dict,
                    tooth_transforms=tooth_transforms,
                    samples=light_samples,
                    neighbor_pairs=light_pairs,
                    clearance=light_cfg.clearance,
                    max_delta_p=light_cfg.max_delta_p,
                    strength=light_cfg.strength,
                )
                step_pose["light_collision"] = {"enabled": True, **light_info}
            else:
                step_pose["light_collision"] = {"enabled": False, "num_pairs_violated": 0, "min_dist": None}

            # apply tooth transforms
            for tooth_id in used_teeth:
                idx = tooth_indices.get(tooth_id, None)
                if idx is None or idx.size == 0:
                    continue

                p = float(p_dict[tooth_id])
                R_full, t_full = tooth_transforms[tooth_id]
                R_p = slerp_rotation(R_full, p)
                t_p = p * t_full
                verts[idx] = (R_p @ verts[idx].T).T + t_p

                step_pose["teeth"][tooth_id] = {
                    "p": float(p),
                    "start_delay": float(start_delay[tooth_id]),
                    "gamma": float(gamma[tooth_id]),
                    "quat_xyzw": Rotation.from_matrix(R_p).as_quat().tolist(),
                    "t_xyz": t_p.tolist(),
                }

            # remove extracted tooth faces after remove_step
            faces = pre_mesh.faces
            if extracted_face_mask is not None and step >= remove_step:
                faces = faces[~extracted_face_mask]

            poses.append(step_pose)

            if export_stl:
                mesh_out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                mesh_out.remove_unreferenced_vertices()
                mesh_out.export(os.path.join(traj_dir, f"step_{step:03d}.stl"))

            if step % max(1, num_steps // 5) == 0:
                print(f"[INFO] traj {k:02d} | step {step:03d}/{num_steps-1:03d} done")

        if save_poses_json:
            meta = {
                "case_dir": case_dir,
                "jaw": jaw,
                "traj_index": k,
                "num_steps": num_steps,
                "num_traj": num_traj,
                "extraction_remove_alpha": extraction_remove_alpha,
                "extraction_remove_step": remove_step,
                "extracted_teeth": extracted_teeth,
                "appearing_teeth": appearing_teeth,
                "schedule_cfg": asdict(sched_cfg),
                "light_collision_cfg": asdict(light_cfg),
                "tooth_report": tooth_report,
                "poses": poses,
            }
            with open(os.path.join(traj_dir, "poses.json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"[DONE] Output saved to: {out_root}")
    return out_root


def _build_argparser():
    import argparse

    p = argparse.ArgumentParser("Pseudo staging generator (Ori -> Final)")
    p.add_argument("--case_dir", type=str, required=True)
    p.add_argument("--jaw", type=str, required=True, choices=["L", "U", "l", "u"])
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--num_traj", type=int, default=8)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--export_stl", action="store_true")
    p.add_argument("--no_export_stl", dest="export_stl", action="store_false")
    p.set_defaults(export_stl=True)

    p.add_argument("--save_poses_json", action="store_true")
    p.add_argument("--no_save_poses_json", dest="save_poses_json", action="store_false")
    p.set_defaults(save_poses_json=True)

    p.add_argument("--extraction_remove_alpha", type=float, default=0.10)

    # schedule
    p.add_argument("--sched_start_max", type=float, default=0.35)
    p.add_argument("--sched_gamma_min", type=float, default=0.8)
    p.add_argument("--sched_gamma_max", type=float, default=1.4)
    p.add_argument("--sched_group_mode", type=str, default="posterior_to_anterior")
    p.add_argument("--sched_group_bonus", type=float, default=0.08)
    p.add_argument("--sched_seed", type=int, default=42)

    # light collision
    p.add_argument("--light_enabled", action="store_true")
    p.add_argument("--light_disabled", dest="light_enabled", action="store_false")
    p.set_defaults(light_enabled=True)
    p.add_argument("--light_clearance", type=float, default=0.002)
    p.add_argument("--light_sample_size", type=int, default=200)
    p.add_argument("--light_neighbor_radius", type=float, default=4.0)
    p.add_argument("--light_max_delta_p", type=float, default=0.08)
    p.add_argument("--light_strength", type=float, default=0.7)
    p.add_argument("--light_seed", type=int, default=0)

    return p


def main():
    args = _build_argparser().parse_args()

    sched_cfg = ScheduleConfig(
        start_max=args.sched_start_max,
        gamma_min=args.sched_gamma_min,
        gamma_max=args.sched_gamma_max,
        group_mode=args.sched_group_mode,
        group_bonus=args.sched_group_bonus,
        seed=args.sched_seed,
    )
    light_cfg = LightColCfg(
        enabled=args.light_enabled,
        clearance=args.light_clearance,
        sample_size=args.light_sample_size,
        neighbor_radius=args.light_neighbor_radius,
        max_delta_p=args.light_max_delta_p,
        strength=args.light_strength,
        seed=args.light_seed,
    )

    generate_multi_pseudo_staging_with_extraction(
        args.case_dir,
        args.jaw,
        num_steps=args.num_steps,
        num_traj=args.num_traj,
        out_dir=args.out_dir,
        export_stl=args.export_stl,
        save_poses_json=args.save_poses_json,
        extraction_remove_alpha=args.extraction_remove_alpha,
        sched_cfg=sched_cfg,
        light_cfg=light_cfg,
    )


if __name__ == "__main__":
    main()
