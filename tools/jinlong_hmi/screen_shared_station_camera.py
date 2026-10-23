#!/usr/bin/env python3
"""Audit the fixed camera for the three-model shared-foot-datum simulation.

Uses archived distortion-aware intrinsics and independent robot/tool ray tests
at the common home. Does not operate hardware or assert actual QR detection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from screen_cycle_camera import evaluate, SIZES, COLOURS
from check_layout_camera import screen as project_outline
from solve_shared_motor_station import SharedStationStudy, argument_parser


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(result, output):
    for path, expected in result["sources_sha256_at_load"].items():
        if result["sources_sha256_at_finish"][path] != expected or digest(path) != expected:
            raise ValueError("Source changed from the validated run: " + path)
    values = result["run_arguments"].copy()
    for action in argument_parser()._actions:
        if action.type is Path and values.get(action.dest) is not None:
            value = values[action.dest]
            values[action.dest] = [Path(v) for v in value] if isinstance(value, list) else Path(value)
    output = output.resolve()
    values["output"] = output
    output.mkdir(parents=True, exist_ok=True)
    study = SharedStationStudy(SimpleNamespace(**values))
    study.station = np.asarray(result["candidate"]["world_T_station"])
    return study


def evaluate_measured_depths(candidate, intr, geometry_files):
    result = evaluate(candidate, intr)
    for fid, row in result["models"].items():
        path = geometry_files[fid]
        geometry = json.loads(path.read_text())
        if "bounds_box_mm" in geometry:
            depth = float(geometry["bounds_box_mm"][1][2])
        else:
            depth = max((float(b["center"][2]) + float(b["half_size"][2])) * 1000
                        for b in geometry["boxes"])
        T = np.asarray(row["camera_T_box"])
        T[:3, 3] += T[:3, 2] * depth / 1000
        half = np.asarray(SIZES[fid]) / 2000
        cavity = project_outline(intr, T, (-half, half), 100, [300, 700])
        row.pop("conservative_90mm_depth_envelope")
        row["conservative_geometry_depth_envelope"] = cavity
        row["geometry_depth_mm"] = depth
        row["geometry_depth_source"] = {"path": str(path), "sha256": digest(path)}
        row["projection_passed"] = (row["rim"]["projection_sample_gates_pass"]
            and cavity["projection_sample_gates_pass"] and row["proposed_QR_patch"]["screen"]["projection_sample_gates_pass"])
    result["all_projection_gates_pass"] = all(r["projection_passed"] for r in result["models"].values())
    return result


def plot(candidate, result, intr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    a, b = axes
    for fid, row in result["models"].items():
        colour = COLOURS[fid]
        T = np.array(candidate["world_T_boxes"][fid])
        h = np.array(SIZES[fid]) / 2
        local = np.array([[-h[0],-h[1]], [h[0],-h[1]], [h[0],h[1]], [-h[0],h[1]], [-h[0],-h[1]]])
        points = np.c_[local, np.zeros(5)] @ T[:3,:3].T + T[:3,3]*1000
        a.plot(points[:,0], points[:,1], color=colour, label=fid+" box")
        route = np.array(candidate["paths"][fid]["outward_flange_poses"])[:,:3,3]*1000
        a.plot(route[:,0], route[:,1], color=colour, ls=":", lw=1)
        for key, style in (("rim", "-"), ("proposed_QR_patch", "--")):
            projection = row[key] if key == "rim" else row[key]["screen"]
            pixels = np.asarray(projection["corner_pixels"])
            pixels = np.vstack([pixels, pixels[0]])
            b.plot(pixels[:,0], pixels[:,1], style, color=colour, label=fid+(" rim" if key=="rim" else " proposed QR"))
    station = np.asarray(candidate["world_T_station"])[:3,3]*1000
    a.plot(*station[:2], "k+", ms=12, label="Common foot locating datum")
    camera = np.asarray(candidate["world_T_camera"])[:3,3]*1000
    a.plot(*camera[:2], "rx", ms=10, label="Fixed camera XY")
    a.set(xlabel="World X (mm)", ylabel="World Y (mm)", title="One shared station; models shown overlaid")
    a.axis("equal"); a.grid(alpha=.2); a.legend(fontsize=8)
    b.set(xlim=(0,intr["width"]), ylim=(intr["height"],0), xlabel="Image u (px)", ylabel="Image v (px)", title="Distorted projection / fixed camera")
    b.set_aspect("equal"); b.grid(alpha=.2); b.legend(fontsize=8)
    fig.suptitle("Common foot datum and table / waist " + str(candidate["waist_model_deg"]) + " deg")
    fig.text(.04,.02,"Each model is loaded separately, with no change of foot datum or table height. QR patch positions are proposals.",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(path,dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--intrinsics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    result=json.loads(a.result.read_text()); candidate=result["candidate"]
    if not result["full_common_home_cycles_verified"]:
        raise ValueError("Requires all three validated cycles at one common home")
    intr=json.loads(a.intrinsics.read_text())
    geometry_files = {}
    for fid in candidate["world_T_boxes"]:
        matches = [Path(p) for p in result["geometry_sha256_at_build"] if Path(p).name == fid + "_collision.json"]
        if len(matches) != 1:
            raise ValueError("Expected one validated box geometry for " + fid)
        if digest(matches[0]) != result["geometry_sha256_at_build"][str(matches[0])]:
            raise ValueError("Box geometry changed")
        geometry_files[fid] = matches[0]
    screen=evaluate_measured_depths(candidate,intr,geometry_files)
    screen["limitations"][1]="One model loaded at the same physical foot datum and table at a time."
    screen["limitations"][4]="Robot/tool occlusion is checked separately at the common home; target self-occlusion and cables are not measured."
    qr_path=a.output/"camera_projection.json"
    qr_path.write_text(json.dumps(screen,indent=2)+"\n")
    study=rebuild(result,a.output/"scenes")
    study.args.qr_layout=qr_path
    boxes={f:np.asarray(t) for f,t in candidate["world_T_boxes"].items()}
    camera=np.asarray(candidate["world_T_camera"]); waist=np.radians(candidate["waist_model_deg"])
    visibility={}
    home=np.asarray(candidate["common_home_validation"]["common_home_arm_rad"])
    for fid in boxes:
        ctx=study.build(fid,boxes,camera,a.output/"scenes"/fid)
        c=dict(candidate); c["path"]={"home_arm_rad":home.tolist()}
        visibility[fid]=study.home_camera_visibility({fid:ctx},waist,c)["fixtures"][fid]
    for path,actual in study.geometry_hashes.items():
        expected=result["geometry_sha256_at_build"].get(path)
        if expected is None and Path(path).name=="rotated_tool_hulls.json":
            suffix="/"+"/".join(Path(path).parts[-2:])
            matches=[v for k,v in result["geometry_sha256_at_build"].items() if k.endswith(suffix)]
            expected=matches[0] if len(matches)==1 else None
        if expected!=actual:
            raise ValueError("Collision input differs from validated result: "+path)
    screen["common_home_visibility"]={"pose":"common home", "fixtures":visibility,
        "all_sample_rays_clear":not any(row["blocked_samples"] for row in visibility.values()),
        "coverage":"85 centre/rim plus 25 proposed-QR rays per motor; robot/tool geometry only"}
    screen["all_camera_geometry_gates_pass"]=screen["all_projection_gates_pass"] and screen["common_home_visibility"]["all_sample_rays_clear"]
    screen["sources_sha256"]={str(path.resolve()):digest(path) for path in (a.result,a.intrinsics,Path(__file__))}
    (a.output/"camera_screen.json").write_text(json.dumps(screen,indent=2)+"\n")
    plot(candidate,screen,intr,a.output/"shared_station_camera.png")
    print(json.dumps({"all_camera_geometry_gates_pass":screen["all_camera_geometry_gates_pass"],
        "blocked_samples":{f:r["blocked_samples"] for f,r in visibility.items()},
        "rim_pixel_margin":{f:r["rim"]["minimum_pixel_border_margin"] for f,r in screen["models"].items()}}))

if __name__=="__main__":
    main()
