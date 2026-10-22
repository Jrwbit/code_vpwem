#!/usr/bin/env python3
"""Solve a saved scene offline with its original geometry and legal checks."""
import argparse
import collections
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--waist", choices=("fixed", "free"), required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    data = json.loads(args.bundle.read_text())
    root, control = Path(data["sim_root"]), Path(data["control_root"])
    sys.path[:0] = [str(control), str(root)]
    from sim.helios_insert import scene as sc, fixtures as fx, corridor as cr, swap as sw, plan
    from helios_hmi.controller_kinematics import apply_collision_model
    from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_start

    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads((args.bundle.parent / "scene_used.json").read_text())
    site, fid = data["site"], data["site"]["selected_fixture"]
    sc.VENDOR = control / "helios_hmi/model/Wheel_Robot"
    configure_geometry(scene, fx, site, root, args.output)
    original = cr.build
    cr.build = lambda cfg: selected_terminal_model(apply_collision_model(original(cfg), site), fid)
    ctx = cr.Ctx(copy.deepcopy(scene), viz=False)
    robot, target = ctx.robot, sc.targets(scene)[0]
    robot.active_motor = fid
    waist, home = planning_start(data)
    robot.set(cr.WAIST, waist)
    robot.set(cr.LEFT, home)
    initial_chassis = robot.get(sc.CHASSIS).copy()
    rng = np.random.default_rng(20261022)
    alo, ahi = cr._bounds(robot, cr.LEFT, .05)
    historical = json.loads((root / "output/20261016-helios-compact/report.json").read_text())["fixtures"][fid]
    seeds = [np.asarray(data["left"]), home, np.radians(historical["q_hover_deg"]),
             np.radians(historical["q_stage_deg"])]
    seeds += list(rng.uniform(alo, ahi, (args.samples, 7)))
    thresholds = {"clear": ctx.cfg.get("final_min_clear_m", cr.MIN_CLEAR), "limit": sw.MIN_LIMIT,
                  "elbow": ctx.cfg["elbow"]["max_rel_m"], "wrist": sw.WRIST_ELLIPSE_MAX}
    counts, combinations = collections.Counter(), collections.Counter()
    best, solutions = [], []
    started = time.monotonic()

    def inspect(q, solver):
        robot.set(cr.LEFT, q)
        robot.fk()
        actual = robot.tcp_pose(ctx.tcp)
        position_error = float(np.linalg.norm(actual[:3, 3] - target.T[:3, 3]))
        angle_error = math.acos(float(np.clip((np.trace(actual[:3, :3].T @ target.T[:3, :3]) - 1) / 2, -1, 1)))
        robot.enter_insertion(fid)
        try:
            quality = sw.quality(robot, ctx.tcp)
        finally:
            robot.leave_insertion()
        failures = [k for k in thresholds if (quality[k] < thresholds[k] if k in ("clear", "limit")
                                               else quality[k] > thresholds[k])]
        if position_error > plan.POS_TOL or angle_error > plan.ROT_TOL:
            failures.append("pose_error")
        w = robot.get(cr.WAIST).copy()
        if any(not robot.lim[n][0] <= v <= robot.lim[n][1] for n, v in zip(cr.WAIST, w)):
            failures.append("waist_limit")
        if not np.array_equal(robot.get(sc.CHASSIS), initial_chassis):
            raise RuntimeError("Offline solver unexpectedly moved chassis")
        row = {"solver": solver, "waist_rad": w.tolist(), "left_rad": np.asarray(q).tolist(),
               "quality": quality, "position_error_mm": position_error * 1000,
               "rotation_error_deg": math.degrees(angle_error), "failures": failures}
        counts["converged"] += 1
        for key in failures:
            counts[key] += 1
        combinations[",".join(failures) or "legal"] += 1
        score = sum(max(thresholds[k] - quality[k], 0) if k in ("clear", "limit")
                    else max(quality[k] - thresholds[k], 0) for k in thresholds)
        best.append((score, row))
        best.sort(key=lambda item: item[0])
        del best[12:]
        if not failures:
            counts["legal"] += 1
            solutions.append(row)
            print("LEGAL", json.dumps(row, default=serial), flush=True)
        return not failures

    if args.waist == "fixed":
        for index, seed in enumerate(seeds):
            for solver in (sw.arm_ik, sw.ik_low_elbow, cr.ik_stage):
                if time.monotonic() - started > args.seconds or solutions:
                    break
                robot.set(cr.WAIST, waist)
                counts["attempts"] += 1
                ok, q = solver(robot, target.T, ctx.tcp, seed)
                if ok:
                    inspect(q, solver.__name__)
            if index % 32 == 0:
                print("PROGRESS", index, dict(counts), flush=True)
            if time.monotonic() - started > args.seconds or solutions:
                break
    else:
        lower = np.array([robot.lim[n][0] + .02 for n in plan.PLACE])
        upper = np.array([robot.lim[n][1] - .02 for n in plan.PLACE])
        for index, seed in enumerate(seeds):
            if time.monotonic() - started > args.seconds or solutions:
                break
            w = (waist if index < 4 else np.clip(waist + rng.normal(0, .35, 4), lower[:4], upper[:4]))
            place = np.r_[w, initial_chassis]
            counts["attempts"] += 1
            ok, solved_place, q = plan.wb_ik(robot, target.T, ctx.tcp, cr.WAIST, place, seed, lower, upper)
            if ok:
                robot.set(cr.WAIST, solved_place[:4])
                if inspect(q, "waist_and_arm_ik"):
                    break
                for solver in (sw.ik_low_elbow, cr.ik_stage):
                    counts["attempts"] += 1
                    good, improved = solver(robot, target.T, ctx.tcp, q)
                    if good and inspect(improved, "free_waist_then_" + solver.__name__):
                        break
            if index % 32 == 0:
                print("PROGRESS", index, dict(counts), flush=True)
    result = {"waist_mode": args.waist, "controller_writes": False, "thresholds": thresholds,
              "target_world_T_flange": target.T.tolist(), "counts": dict(counts),
              "combinations": dict(combinations), "solutions": solutions,
              "closest_rejected": [row for score, row in best], "elapsed_s": time.monotonic() - started,
              "sample_budget": args.samples, "time_budget_s": args.seconds,
              "scope": "endpoint only; approach and full route are not verified"}
    (args.output / "endpoint_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=serial))
    print("FINAL", args.waist, dict(counts), "seconds", result["elapsed_s"], flush=True)


def serial(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
