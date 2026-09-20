"""Create static previews of the two printable split-clamp assemblies."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


ROOT = Path("/data/wjr/vpwem/code_vpwem/cad/robotiq_camera_clamps")


def render(short, title):
    camera = trimesh.load(ROOT / f"robotiq_{short}_split_clamp_camera_half.stl",
                          force="mesh")
    plain = trimesh.load(ROOT / f"robotiq_{short}_split_clamp_plain_half.stl",
                         force="mesh")
    fig = plt.figure(figsize=(11, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for mesh, color in ((camera, "#2e86c1"), (plain, "#75b9e6")):
        poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolor=color,
                                edgecolor="#1c4057", linewidth=0.06)
        ax.add_collection3d(poly)
    all_vertices = np.vstack((camera.vertices, plain.vertices))
    lo, hi = all_vertices.min(axis=0), all_vertices.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(hi - lo) * 0.56
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(ROOT / f"preview_{short}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


render("hande", "Hand-E coupling clamp — D435 / D455 camera mount")
render("ft300s", "FT 300-S coupling clamp — D435 / D455 camera mount")
