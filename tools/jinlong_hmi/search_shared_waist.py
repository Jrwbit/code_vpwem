#!/usr/bin/env python3
"""Search one offline waist posture for measured 132/160 and an estimated 100."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def serial(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle160", type=Path)
    ap.add_argument("vision132", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--height100-from132-mm", type=float, required=True)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--require-approach-mm", type=float, default=0)
    ap.add_argument("--initial-solution", type=Path)
    args = ap.parse_args()
    data = json.loads(args.bundle160.read_text())
    vision = json.loads(args.vision132.read_text())
    if str(vision.get("model_type")) != "132" or not vision.get("success"):
        raise ValueError("132 source must be a successful measured detection")
    if not math.isfinite(args.height100_from132_mm) or args.height100_from132_mm < 0:
        raise ValueError("Estimated 100 height reduction must be nonnegative")
    root, control = Path(data["sim_root"]), Path(data["control_root"])
    sys.path[:0] = [str(control), str(root)]
    from sim.helios_insert import scene as sc, fixtures as fx, corridor as cr, swap as sw, plan
    from helios_hmi.controller_kinematics import apply_collision_model
    from helios_hmi.urdf_kin import UrdfKinematics, default_urdf_path
    from helios_hmi.kinematic_reference import FrameModel
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_start
    from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model

    args.output.mkdir(parents=True, exist_ok=True)
    site = data["site"]
    frames = FrameModel(UrdfKinematics(default_urdf_path()), data["calibration"], site)
    base = json.loads((args.bundle160.parent / "scene_used.json").read_text())
    boxes = {"JM160": frames.world_camera @ np.asarray(data["camera_T_box"]),
             "JL132": frames.world_camera @ np.asarray(vision["camera_T_box"])}
    boxes["JL100"] = boxes["JL132"].copy()
    boxes["JL100"][2, 3] -= args.height100_from132_mm / 1000
    sc.VENDOR = control / "helios_hmi/model/Wheel_Robot"
    original_build = cr.build
    historical = json.loads((root / "output/20261016-helios-compact/report.json").read_text())["fixtures"]
    contexts, goals, seed_cache = {}, {}, {}
    waist0, home = planning_start(data)
    rng = np.random.default_rng(20261022)
    for fixture, box in boxes.items():
        scene = copy.deepcopy(base)
        delta = box @ np.linalg.inv(np.asarray(base["box_T_base_link"]))
        for ob in scene["obstacles"]:
            if ob["name"].startswith("box_") or ob["name"] == "motor_body":
                ob["T"] = (delta @ np.asarray(ob["T"])).tolist()
        scene["box_T_base_link"] = box.tolist()
        scene["terminal_fixture"] = fixture
        case_out = args.output / fixture
        case_out.mkdir(exist_ok=True)
        configure_geometry(scene, fx, site, root, case_out)
        offset = np.asarray(site["tcp_T_objects"][fixture]).copy()
        offset[2, 3] += site.get("fixture_z_adjustments_mm", {}).get(fixture, 0) / 1000
        shift = np.eye(4)
        shift[:3, 3] = np.asarray(site.get("box_center_adjustments_mm", {}).get(fixture, [0, 0, 0])) / 1000
        final = box @ shift @ np.linalg.inv(offset) @ np.linalg.inv(site["flange_T_tcp"])
        scene["targets"] = [{"id": fixture, "tcp": "FLANGE", "T": final.tolist()}]
        cr.build = lambda cfg, fid=fixture: selected_terminal_model(apply_collision_model(original_build(cfg), site), fid)
        ctx = cr.Ctx(copy.deepcopy(scene), viz=False)
        ctx.robot.active_motor = fixture
        for name in cr.WAIST:
            model_lo, model_hi = ctx.robot.lim[name]
            actual_lo, actual_hi = data["limits"][name]
            ctx.robot.lim[name] = (max(model_lo, actual_lo), min(model_hi, actual_hi))
        contexts[fixture], goals[fixture] = ctx, final
        seed_cache[fixture] = [np.radians(historical[fixture]["q_hover_deg"]),
                               np.radians(historical[fixture]["q_stage_deg"]), home, np.asarray(data["left"])]
        (case_out / "scene_used.json").write_text(json.dumps(scene))
    primary = contexts["JM160"].robot
    chassis0 = primary.get(sc.CHASSIS).copy()
    lower = np.array([primary.lim[n][0] + .02 for n in plan.PLACE])
    upper = np.array([primary.lim[n][1] - .02 for n in plan.PLACE])
    alo, ahi = cr._bounds(primary, cr.LEFT, .05)
    started = time.monotonic()
    statistics = {"waist_candidates": 0, "ik_attempts": 0, "partial_counts": {f: 0 for f in boxes},
                  "approach_rejections": {f: 0 for f in boxes}}

    def check(fixture, waist, q, T):
        ctx = contexts[fixture]
        robot = ctx.robot
        robot.set(cr.WAIST, waist)
        robot.set(cr.LEFT, q)
        robot.fk()
        actual = robot.tcp_pose(ctx.tcp)
        dp = float(np.linalg.norm(actual[:3, 3] - T[:3, 3]))
        da = math.acos(float(np.clip((np.trace(actual[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
        robot.enter_insertion(fixture)
        try:
            quality = sw.quality(robot, ctx.tcp)
        finally:
            robot.leave_insertion()
        legal = (sw.legal(quality, ctx.cfg, ctx.cfg.get("final_min_clear_m", cr.MIN_CLEAR))
                 and dp < plan.POS_TOL and da < plan.ROT_TOL
                 and all(robot.lim[n][0] <= v <= robot.lim[n][1] for n, v in zip(cr.WAIST, waist)))
        if not np.array_equal(robot.get(sc.CHASSIS), chassis0):
            raise RuntimeError("Unexpected chassis motion")
        return legal, {"left_rad": np.asarray(q).tolist(), "quality": quality,
                       "position_error_mm": dp * 1000, "rotation_error_deg": math.degrees(da)}

    def approach(fixture, waist, q):
        ctx, T = contexts[fixture], goals[fixture]
        robot = ctx.robot
        current_q = np.asarray(q).copy()
        points = []
        for height in np.linspace(0, args.require_approach_mm / 1000,
                                  max(2, int(math.ceil(args.require_approach_mm / 5)) + 1))[1:]:
            target = sw.raised(T, height)
            ok, solved = sw.arm_ik(robot, target, ctx.tcp, current_q)
            if not ok:
                return None
            legal, row = check(fixture, waist, solved, target)
            if not legal:
                return None
            current_q = np.asarray(solved).copy()
            points.append({"stand_off_mm": height * 1000, **row})
        # The outer hover is ordinary free space, with no mating exemptions.
        quality = sw.quality(robot, ctx.tcp)
        if not sw.legal(quality, ctx.cfg, cr.MIN_CLEAR):
            return None
        try:
            segment = cr.linear_seg(ctx, "offline common approach", waist, q, T, target,
                                    current_q, .5, insert=True, contact_motor=fixture, role="insert")
        except sw.PathError:
            return None
        try:
            for a, b in zip(segment["waypoints"], segment["waypoints"][1:]):
                cr.check_line(ctx, a, b, adjust=False, contact_motor=fixture)
        except sw.PathError:
            return None
        actual_end = np.asarray(segment["waypoints"][-1])[4:]
        legal_end, endpoint = check(fixture, waist, actual_end, target)
        quality = sw.quality(robot, ctx.tcp)
        if not legal_end or not sw.legal(quality, ctx.cfg, cr.MIN_CLEAR):
            return None
        return {"stand_off_mm": args.require_approach_mm, "samples": points,
                "ordinary_hover_quality": quality, "production_cpu_line_passed": True,
                "exported_hover": endpoint,
                "waypoints": [np.asarray(row).tolist() for row in segment["waypoints"]]}

    def solve(fixture, waist, extras=(), random_count=8):
        ctx = contexts[fixture]
        robot, T = ctx.robot, goals[fixture]
        lo, hi = cr._bounds(robot, cr.LEFT, .05)
        seeds = list(extras) + seed_cache[fixture] + list(rng.uniform(lo, hi, (random_count, 7)))
        for seed in seeds:
            for solver in (sw.arm_ik, sw.ik_low_elbow, cr.ik_stage):
                if time.monotonic() - started >= args.seconds:
                    return None
                robot.set(cr.WAIST, waist)
                statistics["ik_attempts"] += 1
                ok, q = solver(robot, T, ctx.tcp, seed)
                if ok:
                    legal, row = check(fixture, waist, q, T)
                    if legal:
                        if args.require_approach_mm > 0:
                            checked = approach(fixture, waist, q)
                            if checked is None:
                                statistics["approach_rejections"][fixture] += 1
                                continue
                            row["approach"] = checked
                        seed_cache[fixture] = [np.asarray(q)] + seed_cache[fixture][:7]
                        statistics["partial_counts"][fixture] += 1
                        return row
        return None

    candidates = [(waist0, [])]
    if args.initial_solution:
        prior = json.loads(args.initial_solution.read_text())["common"]
        candidates.insert(0, (np.asarray(prior["waist_model_rad"]),
                              [np.asarray(prior["solutions"]["JM160"]["left_rad"])]))
        for fixture, row in prior["solutions"].items():
            seed_cache[fixture].insert(0, np.asarray(row["left_rad"]))
    # Optional successful independent endpoint is used only as another seed.
    sibling = args.output.parent / "simulation-free/endpoint_result.json"
    if sibling.exists():
        saved = json.loads(sibling.read_text())
        for row in saved.get("solutions", []):
            candidates.append((np.asarray(row["waist_rad"]), [np.asarray(row["left_rad"])]))
    common = None
    for index in range(args.candidates):
        if time.monotonic() - started >= args.seconds:
            break
        if index < len(candidates):
            waist, extras = candidates[index]
        else:
            anchor = candidates[(index // 4) % len(candidates)][0]
            scale = (.15, .3, .5, .7)[index % 4]
            wseed = np.clip(anchor + rng.normal(0, scale, 4), lower[:4], upper[:4])
            place = np.r_[wseed, chassis0]
            seed = rng.uniform(alo, ahi) if index % 2 else seed_cache["JM160"][0]
            statistics["ik_attempts"] += 1
            ok, solved, q = plan.wb_ik(primary, goals["JM160"], "FLANGE", cr.WAIST, place, seed, lower, upper)
            if not ok:
                continue
            waist, extras = solved[:4], [q]
        statistics["waist_candidates"] += 1
        solutions = {}
        for fixture in ("JM160", "JL132", "JL100"):
            row = solve(fixture, waist, extras if fixture == "JM160" else (), random_count=12)
            if row is None:
                break
            solutions[fixture] = row
        if len(solutions) == 3:
            common = {"waist_model_rad": waist.tolist(), "solutions": solutions}
            print("COMMON_FOUND", json.dumps({"waist_model_rad": common["waist_model_rad"],
                  "fixtures": {f: {k: row[k] for k in ("quality", "position_error_mm", "rotation_error_deg")}
                               for f, row in solutions.items()}}, default=serial), flush=True)
            break
        if index % 8 == 0:
            print("PROGRESS", index, statistics, "seconds", round(time.monotonic() - started, 2), flush=True)
    result = {"common": common, "statistics": statistics, "elapsed_s": time.monotonic() - started,
              "controller_writes": False, "full_route_verified": False,
              "required_approach_mm": args.require_approach_mm,
              "estimated_100": {"source": "132 world pose with world Z reduced", "height_reduction_mm": args.height100_from132_mm,
                                "measured": False, "separate_tool_z_adjustment_mm": site.get("fixture_z_adjustments_mm", {}).get("JL100", 0)},
              "sources": {"bundle160": str(args.bundle160), "vision132": str(args.vision132),
                          "calibration_sha256": data["calibration"]["sha256"]},
              "world_T_boxes": {f: T.tolist() for f, T in boxes.items()},
              "world_T_flange_targets": {f: T.tolist() for f, T in goals.items()}}
    (args.output / "shared_waist_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=serial))
    print("FINAL", "common=" + str(common is not None), statistics, "seconds", result["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
