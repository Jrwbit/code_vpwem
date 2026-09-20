"""Render reproducible previews from the STL exported by FreeCAD."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


OUTPUT_DIR = Path(__file__).resolve().parent
mesh = trimesh.load(OUTPUT_DIR / "ur5_m8_vision_target.stl", force="mesh")


def render(filename, elev, azim, size=(8, 7)):
    fig = plt.figure(figsize=size, dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    faces = mesh.vertices[mesh.faces]
    light = np.array([-0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    diffuse = np.clip(mesh.face_normals @ light, 0.0, 1.0)
    brightness = 0.46 + 0.54 * diffuse
    base_color = np.array([0.91, 0.62, 0.12])
    face_colors = np.column_stack(
        [brightness[:, None] * base_color[None, :], np.ones(len(brightness))]
    )
    poly = Poly3DCollection(
        faces,
        facecolors=face_colors,
        edgecolors="none",
        linewidth=0.0,
        alpha=1.0,
    )
    ax.add_collection3d(poly)
    center = mesh.bounds.mean(axis=0)
    radius = mesh.extents.max() * 0.57
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)


render("ur5_m8_vision_target_preview.png", elev=27, azim=-52)
render("ur5_m8_vision_target_side.png", elev=6, azim=-90, size=(8, 6))
print("watertight:", mesh.is_watertight,
      "winding_consistent:", mesh.is_winding_consistent,
      "components:", len(mesh.split(only_watertight=False)))
print("Preview images exported to:", OUTPUT_DIR)
