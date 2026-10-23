#!/usr/bin/env python3
"""Search one shared motor foot datum while holding the calibrated camera fixed.

Offline projection only. This neither moves hardware nor certifies robot paths,
actual QR decoding, depth completeness, or the unmeasured motor-to-box offset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from check_layout_camera import screen, transform
from screen_cycle_camera import evaluate, SIZES
from screen_shared_station_camera import geometry_depth_mm


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(candidate, intrinsics, depths):
    """Same screen as screen_shared_station_camera, with loaded depths cached."""
    result = evaluate(candidate, intrinsics)
    for fid, row in result["models"].items():
        bottom = np.asarray(row["camera_T_box"])
        bottom[:3, 3] += bottom[:3, 2] * depths[fid] / 1000
        half = np.asarray(SIZES[fid]) / 2000
        cavity = screen(intrinsics, bottom, (-half, half), 100, [300, 700])
        row.pop("conservative_90mm_depth_envelope")
        row["conservative_geometry_depth_envelope"] = cavity
        row["geometry_depth_mm"] = depths[fid]
        row["projection_passed"] = all(r["projection_sample_gates_pass"] for r in (
            row["rim"], cavity, row["proposed_QR_patch"]["screen"]))
    result["all_projection_gates_pass"] = all(r["projection_passed"] for r in result["models"].values())
    result["limitations"][-1] = "The calibrated world camera pose is held exactly fixed throughout this search."
    return result


def measures(projection):
    rows = [r for m in projection["models"].values() for r in (
        m["rim"], m["conservative_geometry_depth_envelope"], m["proposed_QR_patch"]["screen"])]
    margin = min(r.get("minimum_pixel_border_margin", -1e9) for r in rows)
    lo = min(r["depth_range_mm"][0] for r in rows)
    hi = max(r["depth_range_mm"][1] for r in rows)
    return float(margin), [float(lo), float(hi)]


def candidate_at(station_boxes, camera, x_mm, y_mm, z_mm, yaw_deg):
    yaw = np.radians(yaw_deg)
    station = np.eye(4)
    station[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    station[:3, 3] = np.asarray([x_mm, y_mm, z_mm]) / 1000
    return {"world_T_station": station.tolist(), "world_T_camera": camera.tolist(),
            "world_T_boxes": {f: (station @ b).tolist() for f, b in station_boxes.items()},
            "station_yaw_deg": float(yaw_deg)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--camera-audit", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--station-z-mm", type=float, default=500)
    p.add_argument("--x-mm", type=float, nargs=2, default=[250, 450])
    p.add_argument("--y-mm", type=float, nargs=2, default=[-150, 100])
    p.add_argument("--step-mm", type=float, default=5)
    p.add_argument("--yaw-deg", type=float, nargs="+", default=[0])
    a = p.parse_args()
    if a.step_mm <= 0 or any(abs(y) >= 90 for y in a.yaw_deg):
        p.error("Use a positive grid step and yaw strictly within +/-90 degrees (box remains far side).")
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = json.loads(a.result.read_text())
    camera_audit = json.loads(a.camera_audit.read_text())
    camera = transform(camera_audit["world_T_calibrated_camera"])
    intrinsics = json.loads(a.intrinsics.read_text())
    station_boxes = {f: transform(t) for f, t in source["candidate"]["station_T_boxes"].items()}
    geometry_paths = {}
    for fid in station_boxes:
        matches = [Path(n) for n in source["geometry_sha256_at_build"] if Path(n).name == fid + "_collision.json"]
        if len(matches) != 1 or digest(matches[0]) != source["geometry_sha256_at_build"][str(matches[0])]:
            raise ValueError("Missing or changed registered box geometry: " + fid)
        geometry_paths[fid] = matches[0]
    sources = [a.result, a.camera_audit, a.intrinsics, Path(__file__),
               Path(__file__).with_name("check_layout_camera.py"),
               Path(__file__).with_name("screen_cycle_camera.py"),
               Path(__file__).with_name("screen_shared_station_camera.py"), *geometry_paths.values()]
    hashes = {str(path.resolve()): digest(path) for path in sources}
    depths = {f: geometry_depth_mm(json.loads(path.read_text())) for f, path in geometry_paths.items()}
    accepted, tested, by_yaw = [], 0, {}
    for yaw in a.yaw_deg:
        yaw_rows = []
        for x in np.arange(a.x_mm[0], a.x_mm[1] + a.step_mm / 2, a.step_mm):
            for y in np.arange(a.y_mm[0], a.y_mm[1] + a.step_mm / 2, a.step_mm):
                candidate = candidate_at(station_boxes, camera, x, y, a.station_z_mm, yaw)
                projection = project(candidate, intrinsics, depths)
                tested += 1
                if projection["all_projection_gates_pass"]:
                    margin, depth_range = measures(projection)
                    row = {"station_xyz_mm": [float(x), float(y), a.station_z_mm],
                           "yaw_deg": yaw, "minimum_pixel_margin": margin,
                           "optical_depth_range_mm": depth_range}
                    accepted.append(row)
                    yaw_rows.append(row)
        by_yaw[str(yaw)] = {"passing_samples": len(yaw_rows)}
        if yaw_rows:
            xy = np.array([r["station_xyz_mm"][:2] for r in yaw_rows])
            by_yaw[str(yaw)]["passing_grid_xy_bounds_mm"] = [xy.min(axis=0).tolist(), xy.max(axis=0).tolist()]
        print(json.dumps({"yaw_deg": yaw, "tested_so_far": tested,
                          "passing_at_yaw": len(yaw_rows)}), flush=True)
    selected = {}
    if accepted:
        selected["largest_projection_margin"] = max(accepted, key=lambda r: r["minimum_pixel_margin"])
        for threshold in (50, 100, 150, 200):
            rows = [r for r in accepted if r["minimum_pixel_margin"] >= threshold]
            if rows:
                selected["furthest_front_with_" + str(threshold) + "px_margin"] = max(
                    rows, key=lambda r: (r["station_xyz_mm"][0], r["minimum_pixel_margin"]))
    for key, row in selected.items():
        candidate = candidate_at(station_boxes, camera, *row["station_xyz_mm"], row["yaw_deg"])
        projection = project(candidate, intrinsics, depths)
        (a.output / (key + ".json")).write_text(json.dumps(
            {"candidate": candidate, "projection": projection}, indent=2, allow_nan=False) + "\n")
    final_hashes = {path: digest(path) for path in hashes}
    if hashes != final_hashes:
        raise ValueError("A source file changed during the grid search")
    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "camera_relocation_allowed": False, "world_T_camera": camera.tolist(),
           "maximum_camera_transform_change": 0.0,
           "station_z_mm": a.station_z_mm, "common_station_yaw_degrees": a.yaw_deg,
           "grid_xy_bounds_mm": [a.x_mm, a.y_mm], "step_mm": a.step_mm,
           "tested_samples": tested, "passing_samples": len(accepted),
           "by_yaw": by_yaw, "selected": selected, "passing_grid": accepted,
           "geometry_depth_mm": depths, "elapsed_s": time.monotonic() - started,
           "controller_writes": False, "physical_execution_verified": False,
           "robot_reachability_verified": False,
           "limits": ["A grid screen is not a proof for every point inside its bounding rectangle.",
                      "Same station and table for all three models; only one model loaded at a time.",
                      "Conservative box envelope and proposed QR patch use archived intrinsics.",
                      "Robot/tool occlusion, actual QR decoding, depth quality and routes require separate checks.",
                      "Registered station offsets retain the prior unmeasured motor-to-box translation assumption."],
           "source_sha256_at_start": hashes, "source_sha256_at_finish": final_hashes}
    (a.output / "search.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("tested_samples", "passing_samples", "by_yaw", "selected", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
