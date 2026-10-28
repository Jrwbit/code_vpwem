#!/usr/bin/env python3
"""Offline waist screening at immutable measured box poses.

Motor solids are extrapolated from the archived box-to-motor registration.
This does not establish a physical foot registration or permit robot motion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from solve_shared_motor_station import SharedStationStudy, argument_parser
from solve_fixed_waist_cycle import save


MODELS = ("JM160", "JL132", "JL100")


def rigid(value):
    t = np.asarray(value, float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Expected a finite rigid transform in metres")
    return t


def load_observations(path):
    document = json.loads(path.read_text())
    if set(document["models"]) != set(MODELS):
        raise ValueError("Exactly three measured models are required")
    world_camera = rigid(document["world_T_camera"])
    boxes = {}
    sources = {}
    for fid, row in document["models"].items():
        source = (path.parent / row["source"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(fid + ": capture hash changed")
        capture = json.loads(source.read_text())
        if (capture.get("success") is not True
                or str(capture.get("model_type")) != fid.removeprefix("JM").removeprefix("JL")):
            raise ValueError(fid + ": capture failed or model identity differs")
        boxes[fid] = world_camera @ rigid(capture["camera_T_box"])
        sources[str(source)] = digest
    return document, boxes, sources


def measured_scene(nominal, measured_box, local_flange_target, retract_mm):
    """Move box-attached solids only; never move the table, camera or robot."""
    document = copy.deepcopy(nominal)
    observed = rigid(measured_box)
    delta = observed @ np.linalg.inv(rigid(nominal["box_T_base_link"]))
    for obstacle in document["obstacles"]:
        name = obstacle["name"]
        if name.startswith("box_") or name == "motor_body" or name.startswith("motor_fitting_"):
            obstacle["T"] = (delta @ rigid(obstacle["T"])).tolist()
    document["box_T_base_link"] = observed.tolist()
    target = observed @ rigid(local_flange_target)
    target[:3, 3] -= observed[:3, 2] * retract_mm / 1000
    document["targets"] = [{"id": document["terminal_fixture"], "tcp": "FLANGE", "T": target.tolist()}]
    # The historical fixed-station certificate does not apply to measured
    # overrides. Retain provenance without presenting it as this run's proof.
    document["nominal_station_provenance"] = document.pop("shared_motor_station", None)
    document.pop("shared_motor_station_validation", None)
    document["physical_motor_registration_confirmed"] = False
    document["measured_box_screening"] = {
        "box_pose_fixed_input": True,
        "motor_body_placement": "archived box-to-motor transform, unverified on this workpiece",
        "table_and_camera_unchanged": True,
        "physical_execution_verified": False,
    }
    return document, target


def run(args):
    observations, boxes, observation_sources = load_observations(args.observations)
    reference = json.loads(args.reference_run.read_text())
    values = reference["run_arguments"].copy()
    values.update(output=str(args.output.resolve()), seconds=args.seconds,
                  random_seeds=12, bundle=str(args.bundle.resolve()),
                  station_design=str(args.station_design.resolve()),
                  fixed_camera_calibration=str(args.calibration.resolve()), camera_lift_mm=0)
    for option in argument_parser()._actions:
        if option.type is Path and values.get(option.dest) is not None:
            v = values[option.dest]
            values[option.dest] = [Path(x) for x in v] if isinstance(v, list) else Path(v)
    study = SharedStationStudy(SimpleNamespace(**values))
    for source in (Path(__file__), args.observations):
        study.loaded_source_hashes[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    seed_trials = {}
    if args.endpoint_seeds:
        seed_data = json.loads(args.endpoint_seeds.read_text())
        seed_trials = {tuple(t["waist_model_deg"]): t.get("endpoints", {}) for t in seed_data["trials"]}
        study.loaded_source_hashes[str(args.endpoint_seeds.resolve())] = hashlib.sha256(args.endpoint_seeds.read_bytes()).hexdigest()
    if not np.allclose(study.fixed_world_T_camera, observations["world_T_camera"], atol=1e-8):
        raise ValueError("Observation and planning cameras differ")
    calibration_sha = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    if calibration_sha != observations["calibration_sha256"]:
        raise ValueError("Observation calibration differs from the planning calibration")
    study.station = np.asarray(study.station_design["world_T_station"])
    nominal_boxes = study.boxes_for(study.station)
    contexts, targets = {}, {}
    for fid in MODELS:
        nominal = study.build(fid, nominal_boxes, study.fixed_world_T_camera,
                              args.output / "nominal_provenance" / fid)
        scene, target = measured_scene(nominal.cfg, boxes[fid],
                                       study.local_flange_target(fid), study.args.endpoint_retract_mm)
        folder = args.output / "measured_scenes" / fid
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / "scene_used.json", scene)
        ctx = study.cr.Ctx(scene, viz=False)
        ctx.fixture = fid
        ctx.robot.active_motor = fid
        for joint in study.cr.WAIST + study.cr.LEFT:
            lo, hi = ctx.robot.lim[joint]
            lower, upper = study.limits[joint]
            ctx.robot.lim[joint] = (max(lo, lower), min(hi, upper))
        contexts[fid], targets[fid] = ctx, target
    # Search new Cartesian escapes; historical connectors remain only seeds.
    study.station_design.pop("model_route_parameters", None)
    # Historical rows are search seeds only. The connector checks every row
    # and edge against the current measured scenes before accepting a path.
    historical_paths = reference.get("candidate", {}).get("paths", {})
    for fid, historical in historical_paths.items():
        cached = list(study.station_design.get("common_home_connectors", {}).get(fid, []))
        cached.extend(historical.get("outward_waist_arm_rad", []))
        study.station_design.setdefault("common_home_connectors", {})[fid] = cached
    if args.rises_mm is not None:
        study.args.rises_mm = args.rises_mm
    trials = []
    chosen = None
    for degrees in args.waist_deg:
        if time.monotonic() >= study.deadline:
            break
        waist = np.radians(degrees)
        paths, checks, failures, endpoints = {}, {}, {}, {}
        for fid, ctx in contexts.items():
            seeds = seed_trials.get(tuple(degrees), {}).get(fid)
            # Prefer a branch shared with already solved motors. Independent
            # random IK branches can each retract but be hard to connect.
            previous = [np.asarray(s["left_rad"]) for s in endpoints.values()]
            if seeds:
                previous.append(np.asarray(seeds["left_rad"]))
            solution = study.find_ik(ctx, fid, waist, targets[fid], previous=previous, collision=True)
            if solution is None:
                structural = study.find_ik(ctx, fid, waist, targets[fid], collision=False)
                failures[fid] = {"reason": "No collision-free structural endpoint in bounded search",
                                 "structural_solution": structural}
                continue
            endpoints[fid] = solution
            if args.endpoint_only:
                continue
            path, rejected = study.one_path(ctx, fid, waist, targets[fid], solution["left_rad"])
            if path is None:
                failures[fid] = rejected
                continue
            check = study.strict_validate({fid: ctx}, waist, path)
            paths[fid], checks[fid] = path, check
            if not check["passed"]:
                failures[fid] = check
        trial = {"waist_model_deg": degrees, "endpoints": endpoints, "local_paths": paths,
                 "local_checks": checks, "failures": failures}
        if len(checks) == 3 and all(c["passed"] for c in checks.values()):
            trial["common_home_validation"] = study.unify_homes(
                contexts, waist, paths, connector_seconds=args.connector_seconds,
                connector_samples=args.connector_samples)
        trial["full_common_home_cycles_verified"] = trial.get("common_home_validation", {}).get("passed", False)
        trials.append(trial)
        print("MEASURED", degrees, "local", list(checks),
              "endpoints", list(endpoints), "common", trial["full_common_home_cycles_verified"], flush=True)
        save(args.output / "progress.json", trials)
        if trial["full_common_home_cycles_verified"]:
            chosen = trial
            break
    result = {"chosen": chosen, "trials": trials,
              "search_budget_exhausted": time.monotonic() >= study.deadline,
              "world_T_boxes": boxes, "world_T_camera": study.fixed_world_T_camera,
              "calibration_sha256": calibration_sha,
              "observation_sources_sha256": observation_sources,
              "physical_motor_registration_confirmed": False,
              "mating_verified": False, "electrical_test_verified": False,
              "physical_execution_verified": False, "controller_writes": False,
              "limitations": ["Motor-body solids use archived box-to-motor transforms, not independent live foot registration.",
                              "Archived terminal/box solids and contact target registration remain unverified on this capture.",
                              "The common table is retained from the reference model; this is a bounded noncontact planning screen."],
              "sources_sha256_at_load": study.loaded_source_hashes,
              "sources_sha256_at_finish": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                           for p in study.loaded_source_hashes}}
    save(args.output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--station-design", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--endpoint-only", action="store_true")
    p.add_argument("--endpoint-seeds", type=Path)
    p.add_argument("--rises-mm", type=float, nargs="+")
    p.add_argument("--connector-seconds", type=float, default=25)
    p.add_argument("--connector-samples", type=int, default=128)
    p.add_argument("--waist-deg", type=float, nargs=4, action="append", required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
