"""Render reproducible STL previews without requiring a display server."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


OUTPUT_DIR = Path("/home/wjr/code/vpwem/cad/ur5_realsense_mount")
mesh = trimesh.load(OUTPUT_DIR / "ur5_realsense_d435_d455_mount.stl",
                    force="mesh")


def render(filename, elev, azim, size):
    fig = plt.figure(figsize=size, dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    faces = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(
        faces, facecolor="#3f8ed0", edgecolor="#173852",
        linewidth=0.08, alpha=1.0)
    ax.add_collection3d(poly)
    center = mesh.bounds.mean(axis=0)
    radius = mesh.extents.max() * 0.56
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)


render("mount_preview_iso.png", elev=25, azim=-55, size=(9, 7))
render("mount_preview_front.png", elev=0, azim=-90, size=(11, 5))
print("watertight:", mesh.is_watertight,
      "winding_consistent:", mesh.is_winding_consistent,
      "components:", len(mesh.split(only_watertight=False)))
print("Preview images exported to:", OUTPUT_DIR)
