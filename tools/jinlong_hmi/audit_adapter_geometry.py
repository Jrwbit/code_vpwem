#!/usr/bin/env python3
"""Audit archived tool/box geometry with NumPy only; no IK, SDK or robot motion.

The adapter examples prove rigid-body invariance at an exact mating target.
They do not model a physical bracket or establish that its flange pose is reachable.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points(transform, xyz):
    xyz = np.asarray(xyz, float)
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def rotation(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    out = np.eye(4)
    out[:3, :3] = ([[1, 0, 0], [0, c, -s], [0, s, c]] if axis == "x"
                   else [[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return out


def fixture_offset(site, name):
    """Use the archived site, including its current per-fixture height correction."""
    result = np.asarray(site["tcp_T_objects"][name], float).copy()
    result[2, 3] += site.get("fixture_z_adjustments_mm", {}).get(name, 0) / 1000
    return result


def box_shift(site, name):
    out = np.eye(4)
    out[:3, 3] = np.asarray(site.get("box_center_adjustments_mm", {}).get(name, [0, 0, 0])) / 1000
    return out


def pin_wall_gap(pin_box, radius, box_T_wall, size):
    """Exact distance for the archived z-parallel cylinder opposite a wall face.

    Require the face-projection and axial overlap that make the face-distance
    expression exact; this routine does not approximate arbitrary tilted bodies.
    """
    if not np.allclose(box_T_wall[:3, :3], np.eye(3), atol=1e-10):
        raise ValueError("The selected wall must be aligned with the box axes")
    if not np.allclose(pin_box[0, :2], pin_box[1, :2], atol=1e-10):
        raise ValueError("The pin must be parallel to the box Z axis")
    half = np.asarray(size) / 2
    center = box_T_wall[:3, 3]
    pin_z = np.sort(pin_box[:, 2])
    wall_z = center[2] + np.array([-half[2], half[2]])
    overlap = min(pin_z[1], wall_z[1]) - max(pin_z[0], wall_z[0])
    projection_on_face = bool(abs(pin_box[0, 0] - center[0]) <= half[0])
    if overlap <= 0 or not projection_on_face:
        raise ValueError("No axial overlap or face projection; face-distance formula is inapplicable")
    inner_y = center[1] + half[1]
    gap = pin_box[0, 1] - radius - inner_y
    if gap < 0:
        raise ValueError("This audit expects the archived nonintersecting pin/wall pair")
    return {"clearance_mm": float(gap * 1000), "wall_inner_y_mm": float(inner_y * 1000),
            "pin_radius_mm": float(radius * 1000), "pin_centerline_box_mm": (pin_box * 1000).tolist(),
            "pin_z_interval_mm": (pin_z * 1000).tolist(), "wall_z_interval_mm": (wall_z * 1000).tolist(),
            "axial_overlap_mm": float(overlap * 1000), "projection_on_wall_face": projection_on_face}


def audit(evidence_path, scene_path, sim_root):
    evidence = json.loads(evidence_path.read_text())
    scene = json.loads(scene_path.read_text())
    scene_sha = digest(scene_path)
    if scene_sha != evidence["input_hashes"]["scene_used.json"]:
        raise ValueError("Scene SHA256 differs from the archived evidence")
    site = evidence["site"]
    world_box = np.asarray(evidence["world_T_box"], float)
    if not np.allclose(world_box, scene["box_T_base_link"], atol=1e-12):
        raise ValueError("Scene and evidence box transforms differ")
    fx_path = sim_root / "sim/helios_insert/fixtures.py"
    spec = importlib.util.spec_from_file_location("geometry_audit_fixtures", fx_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    # The asset and generated primitives share the historical flange frame.
    # Convert both back to the fixed tool TCP before trying new installations.
    old_install = fixtures.flange_T_tcp()
    current_install = np.asarray(site["flange_T_tcp"], float)
    primitives = fixtures.tool_primitives()
    asset = sim_root / "sim/helios_insert/assets/flange_v2_hulls.json"
    hulls = json.loads(asset.read_text())["hulls"]
    primitive_xyz = np.asarray([p for _, a, b, _ in primitives for p in (a, b)])
    hull_xyz = np.concatenate([np.asarray(h) for h in hulls])
    xyz_tcp = points(np.linalg.inv(old_install), np.concatenate([primitive_xyz, hull_xyz]))
    _, pin_a, pin_b, radius = next(p for p in primitives if p[0] == "tool_pin_JM160")
    pin_tcp = points(np.linalg.inv(old_install), [pin_a, pin_b])
    box_tool = box_shift(site, "JM160") @ np.linalg.inv(fixture_offset(site, "JM160"))
    pin_box = points(box_tool, pin_tcp)
    wall = next(o for o in scene["obstacles"] if o["name"] == "box_wall_ny")
    box_wall = np.linalg.inv(world_box) @ np.asarray(wall["T"])
    pair = pin_wall_gap(pin_box, radius, box_wall, wall["size"])
    target = world_box @ box_tool @ np.linalg.inv(current_install)
    saved_target = next(t["T"] for t in evidence["targets"] if t["id"] == "JM160")
    target_error = float(np.max(np.abs(target - np.asarray(saved_target))))
    if target_error > 1e-10:
        raise ValueError("Recomputed exact JM160 target differs from archived target")

    variants = []
    for axis in ("x", "y"):
        for angle in (-90, 90):
            for translation_mm in (0, 50):
                extra = np.eye(4)
                extra[2, 3] = translation_mm / 1000
                install = current_install @ extra @ rotation(axis, angle)
                deviations = {}
                for fid in site["tcp_T_objects"]:
                    desired_tool = world_box @ box_shift(site, fid) @ np.linalg.inv(fixture_offset(site, fid))
                    baseline = points(desired_tool, xyz_tcp)
                    flange_target = desired_tool @ np.linalg.inv(install)
                    transformed = points(flange_target, points(install, xyz_tcp))
                    deviations[fid] = float(np.max(np.linalg.norm(transformed - baseline, axis=1)) * 1000)
                variants.append({"axis": axis, "angle_deg": angle,
                                 "example_tcp_z_translation_mm": translation_mm,
                                 "flange_T_tcp": install.tolist(), "max_deviation_mm_by_fixture": deviations})

    sensitivity = []
    for outward_mm in range(5):
        changed_wall = box_wall.copy()
        changed_wall[1, 3] -= outward_mm / 1000
        changed = pin_wall_gap(pin_box, radius, changed_wall, wall["size"])
        sensitivity.append({"ny_wall_outward_mm": outward_mm, "clearance_mm": changed["clearance_mm"],
                            "meets_8mm_for_this_pair_only": changed["clearance_mm"] >= 8})

    obstacle_rows = []
    for obstacle in scene["obstacles"]:
        row = {"name": obstacle["name"]}
        if "T" in obstacle:
            row["full_size_mm"] = (np.asarray(obstacle["size"]) * 1000).tolist()
            if obstacle["name"].startswith("box_") or obstacle["name"] == "motor_body":
                local = np.linalg.inv(world_box) @ np.asarray(obstacle["T"])
                row["center_in_box_mm"] = (local[:3, 3] * 1000).tolist()
        elif obstacle["name"] == "table":
            z = np.asarray(obstacle["z_from_floor"]) + scene["floor_z"]
            row.update(full_size_xy_mm=(np.asarray(obstacle["size_xy"]) * 1000).tolist(),
                       center_xy_world_mm=(np.asarray(obstacle["center_xy"]) * 1000).tolist(),
                       z_world_interval_mm=(z * 1000).tolist())
        obstacle_rows.append(row)
    keepout = scene["camera_keepout"]
    source_files = [Path(__file__).resolve(), fx_path, asset,
                    sim_root / "sim/helios_insert/scene.py", sim_root / "sim/helios_insert/plan.py",
                    sim_root / "sim/helios_insert/swap.py", sim_root / "sim/helios_insert/corridor.py"]
    maximum = max(v for row in variants for v in row["max_deviation_mm_by_fixture"].values())
    if maximum > 1e-8:
        raise ValueError("Rigid-body invariance check failed")
    return {
        "scope": "archived exact-target matrix/geometry audit only; no IK or full robot collision calculation",
        "controller_writes": False, "ik_searched": False, "full_route_verified": False,
        "physical_adapter_modelled": False, "execution_authorized": False, "executable": False,
        "evidence_sha256": digest(evidence_path), "scene_sha256": scene_sha,
        "scene_matches_archive": True, "archived_input_hashes": evidence["input_hashes"],
        "archived_source_hashes": evidence["source_hashes"],
        "local_source_sha256": {str(p): digest(p) for p in source_files},
        "numpy_version": np.__version__, "archived_site_geometry": site,
        "exact_target_max_matrix_error": target_error,
        "pin_wall": pair, "single_wall_sensitivity": sensitivity,
        "analysis_threshold_mm": 8,
        "analysis_threshold_source": "Archived final CPU check; swap.py LINE_CLEAR = plan.MIN_CLEAR - 0.002 = 0.008 m",
        "minimum_ny_wall_outward_for_8mm_mm": max(0, 8 - pair["clearance_mm"]),
        "sensitivity_limitation": "Translate only the assumed ny wall outward; not a physical measurement, full-clearance verdict, or permission to reduce margins",
        "obstacles": obstacle_rows,
        "camera_keepout": {"full_size_mm": [*[v * 2000 for v in keepout["half_xy"]], keepout["height"] * 1000],
                           "center_xy_world_mm": [v * 1000 for v in keepout["center_xy"]],
                           "z_bottom_world_mm": keepout["z_bottom"] * 1000,
                           "derived_from": keepout.get("derived_from")},
        "table_calibration": scene.get("table_calibration"),
        "invariance": {
            "formula": "W_T_F = B S C^-1 A^-1; W_p = B S C^-1 A^-1 A p_TCP = B S C^-1 p_TCP",
            "reference_box": "Use supplied JM160 world box for all three algebraic checks; not measured 100/132 targets or reachability tests",
            "composition": "A_new = A_current @ Trans_z(example_mm) @ R_axis(angle)",
            "translation_note": "50 mm is solely a mathematical test value, not an adapter design or recommendation",
            "primitive_count": len(primitives), "primitive_endpoint_count": len(primitive_xyz),
            "hull_count": len(hulls), "hull_vertex_count": len(hull_xyz),
            "tested_point_count_per_fixture": len(xyz_tcp), "max_deviation_mm": maximum,
            "note": "Cylinder radii are unchanged under rigid transforms; endpoints and all hull vertices are checked",
            "variants": variants},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--sim-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    result = audit(args.evidence, args.scene, args.sim_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "pin_wall_gap_mm": result["pin_wall"]["clearance_mm"],
                      "z_overlap_mm": result["pin_wall"]["axial_overlap_mm"],
                      "invariance_max_deviation_mm": result["invariance"]["max_deviation_mm"],
                      "full_route_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
