#!/usr/bin/env python3
"""Offline current-observation hover study using the existing Helios corridor model.

No robot connections or commands. Default to the calibration waist. Failed studies
write diagnostics rather than exporting an unchecked trajectory.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def reproject_calibration(calibration, reference_left_base, working_left_base, waist):
    """A nominal URDF reprojection candidate, never a measured calibration."""
    candidate = copy.deepcopy(calibration)
    candidate["base_T_camera"] = (np.linalg.inv(working_left_base) @ reference_left_base
                                  @ np.asarray(calibration["base_T_camera"])).tolist()
    candidate["waist_model_rad"] = np.asarray(waist, float).tolist()
    candidate["derivation"] = {"kind": "nominal_urdf_waist_reprojection",
                               "source_waist_model_rad": calibration["waist_model_rad"],
                               "physical_application": False}
    candidate["independent_validation"] = {"accepted": False,
        "reason": "腰部变换仅由名义URDF推算，必须在实际工作姿态独立验证后才能使用"}
    candidate.pop("sha256", None)
    candidate.pop("source", None)
    return candidate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root", type=Path, required=True)
    ap.add_argument("--vision-root", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-plan", type=Path, help="Historical joint values as IK seeds only; every new segment is revalidated")
    ap.add_argument("--work-waist-deg", type=float, nargs=4,
                    help="Optional simulated working waist (model degrees); requires reprojection and new validation before any real execution")
    args = ap.parse_args()
    if not 1 <= args.seeds <= 100:
        ap.error("seeds must be between 1 and 100")
    sys.path[:0] = [str(args.sim_root.resolve()), str(args.vision_root.resolve() / "app")]
    from hk_pointcloud_registration.robot.rokae_hover import hover_preview
    from helios_hmi.urdf_kin import UrdfKinematics, default_urdf_path
    from sim.helios_insert import corridor as cr, swap as sw
    from sim.helios_insert import run as presentation
    from sim.helios_insert.scene import WAIST, LEFT, HEAD, RIGHT
    linux_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if linux_font.is_file():
        presentation._FONT = (str(linux_font),)

    bundle = json.loads(args.bundle.read_text())
    preview, cal, site, result, state = (bundle[k] for k in ("preview", "calibration", "site", "result", "state"))
    cal["sha256"] = preview["calibration_sha256"]
    # Admission checks use actual wall time. A later successful study remains a
    # historical simulation when its observation expires; it never executes.
    checked = hover_preview(result, cal, site, state, hover_mm=preview["hover_mm"])
    if not np.allclose(checked["base_T_hover_flange"], preview["base_T_hover_flange"], atol=1e-10):
        raise ValueError("Preview changed since the input bundle was captured")
    doc = json.loads(args.scene.read_text())
    kin = UrdfKinematics(default_urdf_path())
    joints = state["real"]["joints"]
    w0 = np.array([joints[n] for n in WAIST])
    w = w0.copy() if args.work_waist_deg is None else np.radians(args.work_waist_deg)
    a = np.array([joints[n] for n in LEFT])
    base = kin.link_pose("AR5-5_07L_base", dict(zip(WAIST, cal["waist_model_rad"])))
    old_box = np.array(doc["box_T_base_link"])
    new_box = base @ np.array(preview["base_T_box"])
    move_box = new_box @ np.linalg.inv(old_box)
    for obstacle in doc["obstacles"]:
        if obstacle["name"].startswith("box_") or obstacle["name"] == "motor_body":
            obstacle["T"] = (move_box @ np.array(obstacle["T"])).tolist()
    # The table is a world obstacle: moving the observed box never moves it.
    doc["box_T_base_link"] = new_box.tolist()
    doc["calibration_waist_model_rad"] = cal["waist_model_rad"]
    cr.apply_camera(doc, base @ np.array(cal["base_T_camera"]), center="camera")
    doc["tool_mesh"] = str(args.sim_root.resolve() / "sim/helios_insert/assets/flange_v2_hulls.json")
    doc.pop("viz_boxes", None)
    doc["static_pose"] = {"head": [joints[n] for n in HEAD], "right_arm": [joints[n] for n in RIGHT]}
    # Retain the historical model's right-hand calibration board conservatively.
    target = base @ np.array(preview["base_T_hover_flange"])
    contact = target.copy()
    contact[:3, 3] += new_box[:3, 2] * preview["hover_mm"] / 1000
    fid = preview["fixture"]
    doc["targets"] = [{"id": fid, "tcp": "FLANGE", "T": contact.tolist()}]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "scene_used.json").write_text(json.dumps(doc, indent=2))
    if not np.allclose(w, w0):
        working_base = kin.link_pose("AR5-5_07L_base", dict(zip(WAIST, w)))
        candidate = reproject_calibration(cal, base, working_base, w)
        (args.out / "working_calibration_candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2))
    start = {"waist": w0.tolist(), "left_arm": a.tolist(), **doc["static_pose"]}
    ctx = cr.Ctx(copy.deepcopy(doc))
    robot = ctx.robot
    sw._set_start(robot, start)
    start_quality = sw.quality(robot, "FLANGE")
    report = {"task_id": preview["task_id"], "fixture": fid, "controller_writes": False,
              "executable": False, "calibration_sha256": preview["calibration_sha256"],
              "site_sha256": preview["site_sha256"], "waist_locked": bool(np.allclose(w, w0)),
              "work_waist_model_rad": w.tolist(),
              "calibration_reprojection_required": not bool(np.allclose(w, w0)),
              "hover_mm": preview["hover_mm"], "observation_expires_at": preview["expires_at"],
              "input_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
              "scene_assumptions": "Historical table/contact geometry and photo-estimated tool/motor dimensions; current physical obstacles require verification",
              "start_quality": start_quality, "attempts": []}
    rng = np.random.default_rng(17)
    lo, hi = sw._bounds(robot, LEFT, .3)
    seeds = [a] + [rng.uniform(lo, hi) for _ in range(args.seeds)]
    seed_home = None
    if args.seed_plan:
        old_plan = json.loads(args.seed_plan.read_text())
        arm_segments = [s for s in old_plan.get("segments", []) if s.get("group") == "left_arm"]
        if arm_segments:
            historical_seeds = []
            for segment in reversed(arm_segments):
                q = np.asarray(segment["steps"][-1]["target"], float)
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise ValueError("Invalid historical joint seed")
                historical_seeds.append(q)
            seeds = historical_seeds + seeds
            seed_home = np.asarray(arm_segments[0]["steps"][-1]["target"], float)
    homes = None
    if not np.allclose(w, w0):
        homes = cr.find_home(ctx, w0, w, np.random.default_rng(5), n=1200, keep=20)
        for home in [seed_home, a]:
            if home is not None and cr._home_ok(ctx, w0, home, w) and cr.line_ok(ctx, np.r_[w0, home], np.r_[w, home], adjust=False):
                homes.insert(0, (0, home))
        print("Working-waist home candidates:", len(homes), flush=True)
    chosen = None
    for index, seed in enumerate(seeds):
        robot.set(WAIST, w)
        ok, qh = cr.ik_stage(robot, target, "FLANGE", seed)
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        def residual(q):
            robot.set(LEFT, q)
            robot.fk()
            actual = robot.tcp_pose("FLANGE")
            return np.r_[target[:3, 3]-actual[:3, 3],
                         .3*Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]
        error = residual(qh)
        ok = ok and np.linalg.norm(error[:3]) < .0005 and np.linalg.norm(error[3:]) < .0009
        if not ok:
            # The original small-angle IK can stall near a wrist limit. Use a
            # bounded full-rotation residual, then retain every original gate.
            bounds = sw._bounds(robot, LEFT, .055)
            fit = least_squares(residual, np.clip(seed, bounds[0]+1e-6, bounds[1]-1e-6),
                                bounds=bounds, max_nfev=250)
            error = residual(fit.x)
            ok = np.linalg.norm(error[:3]) < .0005 and np.linalg.norm(error[3:]) < .0009
            qh = fit.x
        quality = sw.quality(robot, "FLANGE")
        report["attempts"].append({"seed": index, "ik": bool(ok), "q_rad": qh.tolist(), **quality})
        print("Hover IK", index, bool(ok), quality, flush=True)
        if not ok or not sw.legal(quality, ctx.cfg):
            continue
        for slope in cr.CORRIDOR_SLOPES:
            for degrees in cr.CORRIDOR_DIRS_DEG:
                angle = math.radians(degrees)
                u = np.array([math.sin(angle), math.cos(angle), -slope])
                u /= np.linalg.norm(u)
                rows = cr.march(ctx, w, qh, target, u, .51)
                print("Corridor", index, slope, degrees, "metres", rows[-1][0] if rows else 0, flush=True)
                for distance, path in rows:
                    if distance < .30 or not cr.config_ok(ctx, w, path[-1][4:], clear=.02, adjust=True):
                        continue
                    try:
                        prefix = []
                        if homes is None:
                            link = cr.fewest_connect(ctx, np.r_[w, a], path[-1], rng, allow_start=True, n_samples=120)
                        else:
                            linked = None
                            for _, home in homes[:10]:
                                try:
                                    first = cr.fewest_connect(ctx, np.r_[w0, a], np.r_[w0, home], rng,
                                                             allow_start=True, n_samples=120)
                                    link = cr.fewest_connect(ctx, np.r_[w, home], path[-1], rng, n_samples=120)
                                    cr.check_line(ctx, np.r_[w0, home], np.r_[w, home], adjust=False)
                                    linked = first, home
                                    break
                                except sw.PathError:
                                    continue
                            if linked is None:
                                continue
                            first, home = linked
                            prefix = [cr.joint_seg("先在当前腰姿态下收臂", "left_arm", first),
                                      cr.joint_seg("收臂后调整腰部（仅仿真）", "waist", [np.r_[w0, home], np.r_[w, home]], role="waist")]
                        stage = sw.shifted(target, distance*u)
                        candidate = {"id": fid, "rows_out": path, "q_hover": qh,
                                     "T_H": target, "T_S": stage, "len_m": distance,
                                     "drop_m": float(-distance*u[2])}
                        approach = cr.corridor_in(ctx, candidate, w)
                        segments = prefix + [cr.joint_seg("左侧空旷处调姿", "left_arm", link), approach]
                        chosen = segments
                    except sw.PathError:
                        continue
                    break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            break
    report["simulation_passed"] = chosen is not None
    if chosen:
        report["segments"] = [cr.seg_stats(ctx, seg) for seg in chosen]
        plan = sw.export_plan(robot, ctx.cfg, chosen, start, {"waist": w, "sols": {fid: {"hover_m": preview["hover_mm"]/1000}}},
                              True, args.out / "plan_hover.json")
        plan.update(note="本次视觉观测的非接触悬停候选；未授权执行；需独立定位验收及现场审核整个路径",
                    observation=preview, execution_authorized=False)
        (args.out / "plan_hover.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        frames = sw.replay_frames(robot, ctx.cfg, chosen, start)
        sw.render(robot, ctx.cfg, frames, args.out / "hover.mp4", "当前视觉结果：工装悬停候选（仅仿真）",
                  [f"{fid} | 盒外 {preview['hover_mm']} mm | 任务 {preview['task_id']}",
                   "未执行机器人运动；现场障碍与独立定位精度尚需确认"])
    else:
        sw._set_start(robot, start)
        cr.shot(ctx, args.out / "blocked.png", "悬停规划未通过：未生成可执行轨迹",
                [f"{fid}，盒外 {preview['hover_mm']} mm，腰部锁定：{report['waist_locked']}", "请核对可达性、肘部/腕部限制及现场环境"])
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    print("Simulation passed:", chosen is not None, "Report:", args.out / "report.json", flush=True)
    return 0 if chosen else 2


if __name__ == "__main__":
    raise SystemExit(main())
