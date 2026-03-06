# src/render_trajectory.py
import os
import glob
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def decimate_trimesh_with_open3d(mesh: trimesh.Trimesh, target_tris: int) -> trimesh.Trimesh:
    import open3d as o3d
    if len(mesh.faces) <= target_tris:
        return mesh

    o3m = o3d.geometry.TriangleMesh()
    o3m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3m.triangles = o3d.utility.Vector3iVector(mesh.faces)

    o3m_s = o3m.simplify_quadric_decimation(int(target_tris))
    o3m_s.remove_degenerate_triangles()
    o3m_s.remove_duplicated_triangles()
    o3m_s.remove_duplicated_vertices()
    o3m_s.remove_non_manifold_edges()

    v = np.asarray(o3m_s.vertices)
    f = np.asarray(o3m_s.triangles)
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    out.fix_normals()
    return out


def load_mesh(path, do_decimate=True, use_decimation=True, target_tris=25000):
    m = trimesh.load_mesh(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])

    if do_decimate and use_decimation:
        try:
            m = decimate_trimesh_with_open3d(m, target_tris)
        except Exception as e:
            print("[WARN] decimation failed:", os.path.basename(path), e)
    return m


def set_axes_equal(ax, vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def draw_mesh(ax, mesh, elev=-90, azim=-90):
    # remove previous
    for coll in list(ax.collections):
        coll.remove()

    tris = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(tris, linewidths=0.02, alpha=1.0)
    poly.set_facecolor((0.85, 0.85, 1.0))
    poly.set_edgecolor((0.2, 0.2, 0.2))
    ax.add_collection3d(poly)

    set_axes_equal(ax, mesh.vertices)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


def render_traj(traj_dir: str, out_path: str, fps: int = 6, use_decimation: bool = True, target_tris: int = 25000, elev: int = -90, azim: int = -90):
    stl_files = sorted(glob.glob(os.path.join(traj_dir, "step_*.stl")))
    if len(stl_files) == 0:
        raise FileNotFoundError(f"No step_*.stl found in {traj_dir}")

    meshes = [load_mesh(f, do_decimate=True, use_decimation=use_decimation, target_tris=target_tris) for f in stl_files]
    print("Frames:", len(meshes), "| Faces(first):", len(meshes[0].faces), "| Verts(first):", len(meshes[0].vertices))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, meshes[0], elev=elev, azim=azim)

    def update(i):
        draw_mesh(ax, meshes[i], elev=elev, azim=azim)
        ax.set_title(f"Step {i}", fontsize=12)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(meshes), interval=1000 / fps, blit=False)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
  
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print("[SAVED]", out_path)


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser("Render step_*.stl trajectory to mp4/gif")
    p.add_argument("--traj_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="output .mp4 or .gif")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--use_decimation", action="store_true")
    p.add_argument("--no_decimation", dest="use_decimation", action="store_false")
    p.set_defaults(use_decimation=True)
    p.add_argument("--target_tris", type=int, default=25000)
    p.add_argument("--elev", type=int, default=-90)
    p.add_argument("--azim", type=int, default=-90)
    return p


def main():
    args = _build_argparser().parse_args()
    render_traj(
        traj_dir=args.traj_dir,
        out_path=args.out,
        fps=args.fps,
        use_decimation=args.use_decimation,
        target_tris=args.target_tris,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()
