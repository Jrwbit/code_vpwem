"""Render static previews of the two flange camera plates."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


ROOT = Path("/data/wjr/vpwem/code_vpwem/cad/robotiq_flange_camera_plates")


for key, title in (("hande", "Hand-E flange camera plate"),
                   ("ft300s", "FT 300-S flange camera plate")):
    mesh = trimesh.load(ROOT / f"robotiq_{key}_flange_camera_plate.stl",
                        force="mesh")
    fig = plt.figure(figsize=(11, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],
                          facecolor="#3289c8", edgecolor="#163b5a",
                          linewidth=0.06))
    lo, hi = mesh.bounds
    center = (lo + hi) / 2.0
    radius = max(hi - lo) * 0.56
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=35, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(ROOT / f"preview_{key}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
