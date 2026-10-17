#!/usr/bin/env python3
"""Compare an independently aligned contact pose with a saved camera box pose.

Reads an audit bundle only; never connects to or writes a controller. Translations
in the input matrices are metres. The alignment assumption must be field checked.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        raise ValueError("Expected a finite rigid transform")
    return matrix


def difference(estimated, reference):
    estimated, reference = rigid(estimated), rigid(reference)
    delta = estimated[:3, 3] - reference[:3, 3]
    relative = np.linalg.inv(reference) @ estimated
    return {"estimated_xyz_mm": (estimated[:3, 3] * 1000).tolist(),
            "reference_xyz_mm": (reference[:3, 3] * 1000).tolist(),
            "delta_camera_mm": (delta * 1000).tolist(),
            "delta_box_mm": (reference[:3, :3].T @ delta * 1000).tolist(),
            "position_error_mm": float(np.linalg.norm(delta) * 1000),
            "rotation_error_deg": float(np.degrees(np.arccos(np.clip(
                (np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))),
            "reference_T_estimated": relative.tolist()}


def compare(frames, site, vision, *, fixture=None, gap_mm=0):
    if not np.isfinite(gap_mm) or gap_mm < 0:
        raise ValueError("Gap must be finite and nonnegative")
    fixture = fixture or site["selected_fixture"]
    expected_model = {"JL100": "100", "JL132": "132", "JM160": "160"}[fixture]
    if (str(vision.get("model_type")) != expected_model or vision.get("success") is not True
            or vision.get("center_pose_ready") is not True):
        raise ValueError("A successful matching camera box pose is required")
    current = frames["current"]
    transforms = current["frames"]
    camera_box = rigid(vision["camera_T_box"])
    flange = rigid(transforms["camera_T_left_flange_feedback"])
    nominal = rigid(transforms["camera_T_left_flange"])
    tool = rigid(site["flange_T_tcp"]).copy()
    tcp_z = np.eye(4)
    tcp_z[2, 3] = (site.get("tcp_z_adjustment_mm") or 0) / 1000
    tool = tool @ tcp_z
    offset = rigid(site["tcp_T_objects"][fixture]).copy()
    offset[2, 3] += (site.get("fixture_z_adjustments_mm") or {}).get(fixture, 0) / 1000
    shift = np.eye(4)
    shift[:3, 3] = np.array((site.get("box_center_adjustments_mm") or {}).get(fixture, [0, 0, 0])) / 1000
    gap = np.eye(4)
    gap[2, 3] = gap_mm / 1000
    flange_box = tool @ offset @ np.linalg.inv(shift) @ gap
    inferred = flange @ flange_box
    required = np.linalg.inv(flange) @ camera_box
    waist = np.degrees(current["waist_model_rad"])
    waist[1] -= 90
    return {"fixture": fixture, "gap_mm": gap_mm, "feedback_t": frames.get("feedback_t"),
            "vision_task_id": vision.get("task_id"), "vision_completed_at": vision.get("completed_at"),
            "waist_controller_deg": waist.tolist(), "left_controller_deg": np.degrees(current["left_rad"]).tolist(),
            "sdk_contact_vs_vision": difference(inferred, camera_box),
            "nominal_contact_vs_vision": difference(nominal @ flange_box, camera_box),
            "nominal_vs_sdk_flange": difference(nominal, flange),
            "camera_T_inferred_box": inferred.tolist(),
            "configured_flange_T_box_at_gap": flange_box.tolist(),
            "contact_implied_flange_T_box": required.tolist(),
            "assumption": "Tool physically aligned to the same unchanged box at the stated gap; one pose cannot independently identify calibration and mounting errors"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--vision", default="vision_132.json")
    parser.add_argument("--gap-mm", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.bundle / name).read_text())
    report = compare(read("frames.json"), read("site.json"), read(args.vision), gap_mm=args.gap_mm)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
