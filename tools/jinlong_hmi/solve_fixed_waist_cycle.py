#!/usr/bin/env python3
"""Offline shared-flange workcell search, with separate indexed motor placements.

No hardware client is imported.  The three motors are loaded one at a time.
Their placement transforms are derived from the actual tool mounting transform,
so the robot can reuse one flange orientation and one fixed-waist motion branch.
Results are simulation evidence, not controller commands or electrical acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


FIXTURES = ("JM160", "JL132", "JL100")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform(xyz, yaw_deg):
    t = np.eye(4)
    t[:3, :3] = rz(math.radians(yaw_deg)) @ np.diag([1.0, -1.0, -1.0])
    t[:3, 3] = xyz
    return t


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bounded_contact_horizon(model, required_m=.010, body_padding_m=.020):
    """Reduce near-contact reporting, while proving every enabled pair covered.

    MuJoCo adds both geom margins.  The old tool60 + box30 mm reports contacts
    out to 90 mm; the acceptance rule needs at most 10 mm, plus the existing
    tool/body padding.  This changes neither solids nor acceptance thresholds.
    """
    names = [model.geom(i).name or model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)]
    tool = np.array([n.startswith("tool_") for n in names])
    ct, ca = model.geom_contype, model.geom_conaffinity
    old = model.geom_margin.copy()
    new = old.copy()
    for i in range(model.ngeom):
        if tool[i]:
            new[i] = 0.0
        elif ct[i] & 1:
            new[i] = required_m + body_padding_m + .0005
        elif ct[i] == 2:
            new[i] = required_m + .0005
        elif ct[i] == 4:
            new[i] = .005
    enabled = ((ct[:, None] & ca[None, :]) != 0) | ((ca[:, None] & ct[None, :]) != 0)
    body = (ct & 1) != 0
    padded = (tool[:, None] ^ tool[None, :]) & ((tool[:, None] & body[None, :]) | (body[:, None] & tool[None, :]))
    requirement = required_m + padded * body_padding_m
    horizon = new[:, None] + new[None, :]
    reserve = horizon - requirement
    if model.npair:
        raise ValueError("Explicit MuJoCo contact pairs need their own horizon audit")
    if np.any(reserve[enabled] < 0):
        raise ValueError("Reduced contact horizon would omit a required collision pair")
    model.geom_margin[:] = new
    return {"required_clearance_including_generation_buffer_mm": required_m * 1000,
            "existing_tool_body_padding_mm": body_padding_m * 1000,
            "enabled_pair_combinations_checked": int(np.count_nonzero(enabled)),
            "minimum_coverage_reserve_mm": float(reserve[enabled].min() * 1000),
            "unreported_effective_clearance_lower_bound_mm": float((horizon - padded * body_padding_m)[enabled].min() * 1000),
            "geometry_margins_mm": [{"name": name, "old": float(a * 1000), "new": float(b * 1000)}
                                    for name, a, b in zip(names, old, new)],
            "solids_and_acceptance_thresholds_unchanged": True,
            "reported_minimum_is_censored": True}


class Study:
    def __init__(self, args):
        self.args = args
        self.design = json.loads(args.design_config.read_text()) if args.design_config else None
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path[:0] = list(map(str, (args.control_root, args.sim_root, args.vision_root)))
        from sim.helios_insert import scene, fixtures, corridor, swap, plan
        from helios_hmi.controller_kinematics import apply_collision_model
        from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_limits
        from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model
        self.sc, self.fx, self.cr, self.sw, self.plan = scene, fixtures, corridor, swap, plan
        self.apply_model, self.configure = apply_collision_model, configure_geometry
        self.select = selected_terminal_model
        scene.VENDOR = args.control_root / "helios_hmi/model/Wheel_Robot"
        self.original_build = corridor.build
        self.data = json.loads(args.bundle.read_text())
        self.base = json.loads(args.scene.read_text())
        self.site = self.data["site"]
        self.limits = planning_limits(self.data["limits"], self.site)
        self.rng = np.random.default_rng(args.seed)
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.log = []
        self.seeds = []
        for path in args.seed_results:
            old = json.loads(path.read_text())
            chosen = old.get("chosen")
            if chosen:
                for sol in chosen["solutions"].values():
                    self.seeds.extend([np.array(sol["final"]["left_rad"]), np.array(sol["pre30"]["left_rad"])])
        self.seeds += [np.array(self.data["left"]),
                       np.array([-1.432826609, -2.007128640, 1.316061868, .497740107, 1.189695716, .895000094, .587892953])]
        if self.design and self.design.get("seed_left_rad"):
            self.seeds.insert(0, np.asarray(self.design["seed_left_rad"], float))
        self.adjusted = {}
        self.shifts = {}
        for fid in FIXTURES:
            c = np.array(self.site["tcp_T_objects"][fid])
            c[2, 3] += self.site.get("fixture_z_adjustments_mm", {}).get(fid, 0) / 1000
            s = np.eye(4)
            s[:3, 3] = np.array(self.site.get("box_center_adjustments_mm", {}).get(fid, [0, 0, 0])) / 1000
            self.adjusted[fid], self.shifts[fid] = c, s
        self.mount = np.array(self.site["flange_T_tcp"])
        self.contact_offsets = {f: 0.0 for f in FIXTURES}
        if args.contact_offsets:
            self.contact_offsets.update(json.loads(args.contact_offsets.read_text()))
        self.target_corrections = json.loads(args.target_corrections.read_text()) if args.target_corrections else {}
        self.fixture_revision = json.loads(args.fixture_revision.read_text()) if args.fixture_revision else None
        self.original_box_flange = {f: self.shifts[f] @ np.linalg.inv(self.adjusted[f]) @ np.linalg.inv(self.mount)
                                    for f in FIXTURES}
        expected = np.eye(4); expected[:3, :3] = rz(-math.pi / 2)
        if not np.allclose(self.mount, expected):
            raise ValueError("Study requires the verified Rz(-90 degree) tool mounting")
        self.geometry = load_module(args.geometry_module, "cycle_geometry") if args.geometry_module else None
        self.validator = load_module(args.validation_module, "cycle_validator") if args.validation_module else None
        self.tool_geometry = load_module(args.tool_geometry_module, "cycle_tool_geometry") if args.tool_geometry_module else None
        self.loaded_source_hashes = {}
        self.geometry_hashes = {}
        self.horizon_proofs = {}
        input_paths = [p for p in (args.contact_offsets, args.target_corrections, args.fixture_revision,
                                  args.design_config, args.qr_layout) if p]
        for path in [Path(__file__), args.bundle, args.scene] + input_paths + [Path(m.__file__) for m in
                (self.sc, self.fx, self.cr, self.sw, self.plan, self.geometry, self.validator, self.tool_geometry) if m]:
            self.loaded_source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def design_boxes(self, flange):
        """All boxes put the selected tool at this same nominal flange pose."""
        return {f: flange @ np.linalg.inv(self.local_flange_target(f)) for f in FIXTURES}

    def local_flange_target(self, fid):
        correction = np.eye(4)
        values = self.target_corrections.get(fid, {})
        correction[:3, :3] = rz(math.radians(values.get("yaw_deg", 0)))
        correction[:3, 3] = np.asarray(values.get("translation_mm", [0, 0, 0])) / 1000
        local = correction @ self.original_box_flange[fid]
        local[2, 3] += self.contact_offsets[fid] / 1000
        return local

    def camera_pose(self, boxes):
        if self.design and self.design.get("world_T_camera") is not None:
            return np.asarray(self.design["world_T_camera"], float)
        points = np.array([t[:3, 3] for t in boxes.values()])
        t = np.eye(4)
        # Optical +Z down; image +X world +X, image +Y world -Y.
        t[:3, :3] = np.diag([1.0, -1.0, -1.0])
        t[:2, 3] = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
        t[2, 3] = points[:, 2].max() + self.args.camera_height_mm / 1000
        return t

    def build(self, fid, boxes, camera, folder):
        folder.mkdir(parents=True, exist_ok=True)
        doc = copy.deepcopy(self.base)
        doc.pop("viz_boxes", None)
        old_box = np.array(self.base["box_T_base_link"])
        delta = boxes[fid] @ np.linalg.inv(old_box)
        all_points = np.array([b[:3, 3] for b in boxes.values()])
        for ob in doc["obstacles"]:
            if ob["name"].startswith("box_") or ob["name"] == "motor_body":
                ob["T"] = (delta @ np.array(ob["T"])).tolist()
            elif ob["name"] == "table":
                # One common large table, not a table moved independently for each motor.
                ob["center_xy"] = [float(all_points[:, 0].mean()) + .10, float(all_points[:, 1].mean())]
                tabletop = float(all_points[:, 2].min()) - .40
                ob["z_from_floor"] = [tabletop - doc["floor_z"] - .06, tabletop - doc["floor_z"]]
        motor = next(o for o in doc["obstacles"] if o["name"] == "motor_body")
        motor_pose = np.asarray(motor["T"])
        half = np.asarray(motor["size"]) / 2
        corners = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
                            for z in (-half[2], half[2])]) @ motor_pose[:3, :3].T + motor_pose[:3, 3]
        bottom = float(corners[:, 2].min())
        if bottom > tabletop:
            support = np.eye(4)
            support[:2, 3] = (corners[:, :2].min(0) + corners[:, :2].max(0)) / 2
            support[2, 3] = (bottom + tabletop) / 2
            doc["obstacles"].append({"name": "motor_support", "T": support.tolist(),
                                      "size": [*np.ptp(corners[:, :2], axis=0), bottom - tabletop],
                                      "rgba": [.35, .38, .42, 1],
                                      "design_note": "Conservative support envelope, not manufactured riser CAD"})
        doc["box_T_base_link"] = boxes[fid].tolist()
        self.cr.apply_camera(doc, camera, center="camera", half_xy=.35, below_lens=.05)
        doc["terminal_fixture"] = fid
        doc["left_arm_soft_limits_rad"] = {"lower": [self.limits[n][0] for n in self.sc.LEFT],
                                           "upper": [self.limits[n][1] for n in self.sc.LEFT]}
        doc["static_pose"] = self.data["static_pose"]
        self.configure(doc, self.fx, self.site, self.args.sim_root, folder)
        target = boxes[fid] @ self.local_flange_target(fid)
        target[:3, 3] -= target[:3, 2] * self.args.endpoint_retract_mm / 1000
        doc["targets"] = [{"id": fid, "tcp": "FLANGE", "T": target.tolist()}]
        use_geometry = self.geometry is not None and fid in self.args.geometry_fixtures
        if use_geometry:
            self.geometry.modify_scene(doc, fid)
        if self.tool_geometry:
            if not use_geometry:
                raise ValueError("Tool CAD hook requires the model-specific builder for all selected fixtures")
            self.tool_geometry.modify_scene(doc, self.mount)
        if self.fixture_revision:
            doc["tool_fixture_revision"] = self.fixture_revision
            doc["tool_fixture_reference_box_T_flange"] = {f: t.tolist() for f, t in self.original_box_flange.items()}
        for key in ("model_specific_box_collision", "tool_cad_collision", "tool_mesh"):
            if doc.get(key):
                path = Path(doc[key])
                self.geometry_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        def build_model(cfg):
            spec = (self.geometry.build_model(cfg, fid, self.original_build)
                    if use_geometry else self.original_build(cfg))
            model = self.select(self.apply_model(spec, self.site), fid)
            if self.args.bounded_contact_horizon:
                self.horizon_proofs[fid] = bounded_contact_horizon(model)
            return model
        self.cr.build = build_model
        ctx = self.cr.Ctx(copy.deepcopy(doc), viz=False)
        ctx.fixture = fid
        ctx.robot.active_motor = fid
        for n in self.cr.WAIST + self.cr.LEFT:
            a, b = ctx.robot.lim[n]; c, d = self.limits[n]
            ctx.robot.lim[n] = (max(a, c), min(b, d))
        save(folder / "scene_used.json", doc)
        return ctx

    def quality(self, ctx, waist, q, target=None, contact=False):
        robot = ctx.robot
        robot.set(self.cr.WAIST, waist); robot.set(self.cr.LEFT, q); robot.fk()
        if contact:
            robot.enter_insertion(ctx.fixture)
        try:
            quality = self.sw.quality(robot, "FLANGE")
        finally:
            robot.leave_insertion()
        if self.args.bounded_contact_horizon:
            lower_bound = self.horizon_proofs[ctx.fixture]["unreported_effective_clearance_lower_bound_mm"] / 1000
            quality["minimum_clearance_is_lower_bound"] = quality["clear"] >= lower_bound
            # Robot.clearance() returns its legacy 30 mm cap when nothing is
            # reported. After narrowing the reporting horizon only this lower
            # bound is established; do not turn absent contacts into 30 mm.
            quality["clear"] = min(quality["clear"], lower_bound)
        actual = robot.tcp_pose("FLANGE")
        if target is not None:
            quality["position_error_mm"] = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000)
            quality["rotation_error_deg"] = math.degrees(float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))))
        quality["structural_ok"] = (quality["limit"] >= self.plan.MIN_LIMIT and
                                    quality["wrist"] <= self.sw.WRIST_ELLIPSE_MAX and
                                    quality["elbow"] <= ctx.cfg["elbow"]["max_rel_m"])
        return quality

    def ik(self, ctx, waist, target, seeds, low_elbow=False):
        r = ctx.robot
        r.set(self.cr.WAIST, waist)
        solver = self.sw.ik_low_elbow if low_elbow else self.sw.arm_ik
        for seed in seeds:
            ok, q = solver(r, target, "FLANGE", seed)
            if ok:
                quality = self.quality(ctx, waist, q, target)
                if quality["structural_ok"]:
                    return np.array(q), quality
        return None, None

    def trace(self, contexts, waist, initial_q, poses, contact_from=-1):
        ctx = contexts[FIXTURES[0]]
        q = np.array(initial_q)
        rows = [q.copy()]
        records = []
        for index, target in enumerate(poses):
            qnext, quality = self.ik(ctx, waist, target, [q])
            if qnext is None:
                qnext, quality = self.ik(ctx, waist, target, [q], low_elbow=True)
            if qnext is None:
                return None, {"failed_index": index, "reason": "IK/structural", "pose": target.tolist()}
            # Keep continuation continuous; do not hide null-space branch jumps.
            if np.abs(qnext - q).max() > math.radians(12):
                return None, {"failed_index": index, "reason": "joint discontinuity"}
            qualities = {}
            for fid, c in contexts.items():
                qual = self.quality(c, waist, qnext, target, contact=index >= contact_from >= 0)
                qualities[fid] = qual
                minimum = 0.0 if index >= contact_from >= 0 else .008
                if not qual["structural_ok"] or qual["clear"] < minimum:
                    return None, {"failed_index": index, "reason": "scene/structural", "fixture": fid, "quality": qual}
            rows.append(qnext.copy()); q = qnext
            records.append(qualities)
        return rows, {"waypoint_quality": records}

    def path_candidate(self, contexts, waist, flange, qfinal):
        # Solve from final outward.  Each pose is later checked densely in both directions.
        tried = []
        for rise_mm in self.args.rises_mm:
            for azimuth in self.args.escape_azimuths_deg:
                if time.monotonic() >= self.deadline:
                    return None, tried
                rise = rise_mm / 1000
                retreat = []
                for h in np.arange(.005, rise + .0001, .005):
                    t = flange.copy(); t[2, 3] += h; retreat.append(t)
                # Initial 30 mm retains the legacy insertion allowance only for exploration.
                q = np.array(qfinal)
                armrows = [q.copy()]
                failure = None
                allquality = []
                for j, t in enumerate(retreat):
                    qnext, quality = self.ik(contexts["JM160"], waist, t, [q])
                    if qnext is None:
                        qnext, quality = self.ik(contexts["JM160"], waist, t, [q], low_elbow=True)
                    if qnext is None or np.abs(qnext - q).max() > math.radians(12):
                        failure = {"stage": "retreat", "index": j, "reason": "IK/structural/discontinuous"}; break
                    qualities = {}
                    for fid, ctx in contexts.items():
                        qual = self.quality(ctx, waist, qnext, t, contact=j < 6)
                        qualities[fid] = qual
                        if qual["clear"] < (0.0 if j < 6 else .008):
                            failure = {"stage": "retreat", "index": j, "fixture": fid, "quality": qual}; break
                    if failure:
                        break
                    armrows.append(qnext.copy()); q = qnext; allquality.append(qualities)
                if failure:
                    tried.append(failure); break  # azimuth cannot change a vertical failure
                direction = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth)), 0.0])
                poses = []
                start = retreat[-1]
                for distance in np.arange(.01, self.args.escape_mm / 1000 + .0001, .01):
                    t = start.copy(); t[:3, 3] += direction * distance; poses.append(t)
                lateral, result = self.trace(contexts, waist, q, poses)
                if lateral is None:
                    tried.append({"rise_mm": rise_mm, "azimuth": azimuth, "stage": "lateral", **result}); continue
                armrows += lateral[1:]
                transforms = [flange] + retreat + poses
                # A home away from the workpiece is allowed to be chosen freely by the user.
                # It need not satisfy adjust-zone rules if every cycle retains tool orientation.
                return {"rise_mm": rise_mm, "escape_azimuth_deg": azimuth,
                        "outward_arm_rad": armrows, "outward_flange_poses": transforms,
                        "home_arm_rad": armrows[-1], "home_flange_pose": transforms[-1],
                        "outward_waist_arm_rad": [np.r_[waist, a] for a in armrows],
                        "exploration_quality": allquality}, tried
        return None, tried

    def strict_validate(self, contexts, waist, path):
        if not self.validator:
            return {"passed": False, "reason": "strict validation module not provided"}
        rows = path["outward_waist_arm_rad"]
        poses = path["outward_flange_poses"]
        results = {}
        segments = []
        # Execute home -> endpoint -> the identical home, preserving every seam.
        for direction in (-1, 1):
            indices = range(len(rows) - 2, -1, -1) if direction == -1 else range(len(rows) - 1)
            for i in indices:
                a, b = (i + 1, i) if direction == -1 else (i, i + 1)
                distance = min(np.linalg.norm(np.array(poses[a])[:3, 3] - np.array(poses[0])[:3, 3]),
                               np.linalg.norm(np.array(poses[b])[:3, 3] - np.array(poses[0])[:3, 3]))
                phase = (("retract" if direction == 1 else "insert")
                         if distance <= .03001 and self.args.endpoint_retract_mm == 0 else "corridor")
                segments.append({"name": ("return_" if direction == 1 else "approach_") + str(i),
                                 "kind": "linear", "phase": phase, "waypoints": [rows[a], rows[b]],
                                 "Ta": poses[a], "Tb": poses[b],
                                 "dev_mm": .5 if distance <= .01001 else 2.0, "angle_deg": 1.0})
        for fid, ctx in contexts.items():
            begin = time.monotonic()
            # Reject an invalid mating end before spending time on its remote corridor.
            first = segments[len(rows) - 2]
            endpoint_check = self.validator.check_segment(ctx, first["waypoints"][0], first["waypoints"][1],
                                                          first["phase"], fid, Ta=first["Ta"], Tb=first["Tb"],
                                                          dev_mm=first["dev_mm"], angle_deg=1.0, waist=waist)
            if not endpoint_check["ok"]:
                result = {"ok": False, "endpoint_check": endpoint_check, "full_cycle_checked": False}
            else:
                result = self.validator.validate_cycle(ctx, segments, fid, waist)
                result["full_cycle_checked"] = True
            result["elapsed_s"] = time.monotonic() - begin
            results[fid] = result
            print("VERIFY", fid, result["ok"], "seconds", round(result["elapsed_s"], 3), flush=True)
        return {"passed": all(r["ok"] for r in results.values()), "fixtures": results,
                "closed_cycle_waypoints": len(rows) * 2 - 1, "one_way_segments": len(rows) - 1}

    def home_camera_visibility(self, contexts, waist, candidate):
        """Ray-check robot/tool occlusion at home, excluding camera/workpiece."""
        import mujoco
        camera = np.asarray(candidate["world_T_camera"], float)
        point = camera[:3, 3].copy()
        results = {}
        for fid, ctx in contexts.items():
            robot = ctx.robot
            robot.set(self.cr.WAIST, waist)
            robot.set(self.cr.LEFT, candidate["path"]["home_arm_rad"])
            robot.fk()
            geometry = json.loads(Path(ctx.cfg["model_specific_box_collision"]).read_text())
            if "bounds_box_mm" in geometry:
                bounds = np.asarray(geometry["bounds_box_mm"], float)[:, :2] / 1000
            else:
                minima = [np.asarray(b["center"]) - np.asarray(b["half_size"])
                          for b in geometry.get("boxes", [])]
                maxima = [np.asarray(b["center"]) + np.asarray(b["half_size"])
                          for b in geometry.get("boxes", [])]
                for part in geometry.get("parts", []):
                    vertices = np.asarray(part["vertices"], float)
                    minima.append(vertices.min(0)); maxima.append(vertices.max(0))
                if not minima:
                    raise ValueError("Camera visibility requires nonempty registered box geometry")
                lo, hi = np.asarray(minima), np.asarray(maxima)
                bounds = np.array([lo.min(0)[:2], hi.max(0)[:2]])
            samples = [[0., 0., 0.]]
            labels = ["rim_center"]
            for fraction in np.linspace(0., 1., 21):
                x = bounds[0, 0] + fraction * (bounds[1, 0] - bounds[0, 0])
                y = bounds[0, 1] + fraction * (bounds[1, 1] - bounds[0, 1])
                samples.extend([[x, bounds[0, 1], 0.], [x, bounds[1, 1], 0.],
                                [bounds[0, 0], y, 0.], [bounds[1, 0], y, 0.]])
                labels.extend(["rim"] * 4)
            if self.args.qr_layout:
                qr = json.loads(self.args.qr_layout.read_text())["models"][fid]["proposed_QR_patch"]
                qr_bounds = np.asarray(qr["bounds_box_mm"], float) / 1000
                for x in np.linspace(qr_bounds[0, 0], qr_bounds[1, 0], 5):
                    for y in np.linspace(qr_bounds[0, 1], qr_bounds[1, 1], 5):
                        samples.append([x, y, 0.]); labels.append("proposed_qr")
            box = np.asarray(candidate["world_T_boxes"][fid], float)
            world = np.asarray(samples) @ box[:3, :3].T + box[:3, 3]
            old_groups = robot.m.geom_group.copy()
            included = []
            try:
                for gid in range(robot.m.ngeom):
                    name = robot._gname(gid)
                    bodyname = robot.m.body(robot.m.geom_bodyid[gid]).name
                    is_robot = (robot.m.geom_bodyid[gid] != 0 and not bodyname.startswith("box_")
                                and not name.startswith(("box_", "term_", "hik_camera", "keepout_")))
                    robot.m.geom_group[gid] = 0 if is_robot else 5
                    if is_robot:
                        included.append(name)
                records = []
                for index, target in enumerate(world):
                    delta = target - point
                    length = float(np.linalg.norm(delta))
                    hit = np.array([-1], dtype=np.int32)
                    distance = float(mujoco.mj_ray(robot.m, robot.d, point, delta / length,
                                                  np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, hit))
                    blocked = 0 <= distance < length - .001
                    records.append({"sample_index": index, "target_world_m": target.tolist(),
                                    "sample_role": labels[index],
                                    "target_distance_m": length, "blocked": blocked,
                                    "first_hit_m": distance if distance >= 0 else None,
                                    "first_hit_geom": robot._gname(int(hit[0])) if hit[0] >= 0 else None})
            finally:
                robot.m.geom_group[:] = old_groups
            results[fid] = {"samples": len(records), "blocked_samples": sum(r["blocked"] for r in records),
                            "center_blocked": records[0]["blocked"], "records": records,
                            "qr_samples": sum(r["sample_role"] == "proposed_qr" for r in records),
                            "blocked_qr_samples": sum(r["sample_role"] == "proposed_qr" and r["blocked"] for r in records),
                            "checked_robot_tool_geometry": included,
                            "bounds_box_m": bounds.tolist()}
        return {"pose": "common home", "fixtures": results,
                "all_sample_rays_clear": not any(r["blocked_samples"] for r in results.values()),
                "coverage": "85 center/outer-rim rays and optional 25 proposed-QR rays per motor; robot/tool geometry only",
                "limitations": "Sampled visibility, not QR detection, image coverage, cables, or depth-quality validation"}

    def mating_metrics(self, contexts, waist, q):
        """Report actual loaded terminal/tool geometry; no pose-only acceptance."""
        import mujoco
        results = {}
        for fid, ctx in contexts.items():
            r = ctx.robot
            r.set(self.cr.WAIST, waist); r.set(self.cr.LEFT, q); r.fk()
            normal = np.asarray(ctx.cfg["box_T_base_link"])[:3, 2]
            pairs = [(f"tool_sock_{fid}_{k}", f"term_{fid}_stud_{k}", "socket") for k in (1, 2, 3)]
            pairs.append((f"tool_pin_{fid}", f"term_{fid}_gnd", "ground"))
            measurements = []
            for tool, terminal, kind in pairs:
                ids = [mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in (tool, terminal)]
                if min(ids) < 0:
                    measurements.append({"tool": tool, "terminal": terminal, "available": False}); continue
                endpoints = []
                for gid in ids:
                    if r.m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                        raise ValueError("Terminal contact metrics require the declared cylinder geometry")
                    axis = r.d.geom_xmat[gid].reshape(3, 3)[:, 2]
                    if axis @ normal < 0:
                        axis = -axis
                    center, half = r.d.geom_xpos[gid], r.m.geom_size[gid, 1]
                    endpoints.append((center - axis * half, center + axis * half, axis))
                root, tip, axis = endpoints[0]
                top, _, t_axis = endpoints[1]
                delta = tip - top
                overlap = float(delta @ t_axis)
                lateral = float(np.linalg.norm(delta - overlap * t_axis))
                angle = math.acos(float(np.clip(axis @ t_axis, -1, 1)))
                radial_room = .0035 - r.m.geom_size[ids[1], 0] if kind == "socket" else r.m.geom_size[ids[1], 0] - r.m.geom_size[ids[0], 0]
                occupied = lateral + max(overlap, 0) * math.sin(angle)
                axis_fit = occupied <= radial_room + 1e-6 and angle <= math.radians(1)
                axial_fit = (-1e-6 <= overlap <= (.002 if kind == "ground" else np.linalg.norm(tip - root)) + 1e-6)
                measurements.append({"tool": tool, "terminal": terminal, "kind": kind, "available": True,
                                     "lateral_error_mm": lateral * 1000, "alignment_deg": math.degrees(angle),
                                     "engagement_or_compression_mm": overlap * 1000,
                                     "modeled_radial_room_mm": float(radial_room * 1000),
                                     "axis_fit": bool(axis_fit), "axial_fit": bool(axial_fit),
                                     "modeled_contact_fit": bool(axis_fit and axial_fit)})
            geometry = json.loads(Path(ctx.cfg["model_specific_box_collision"]).read_text()) if ctx.cfg.get("model_specific_box_collision") else {}
            results[fid] = {"pairs": measurements,
                            "all_four_loaded_pairs_fit": all(m.get("modeled_contact_fit", False) for m in measurements),
                            "workpiece_contacts_independently_measured": bool(geometry.get("measured_terminals")),
                            "minimum_socket_engagement_requirement_confirmed": False,
                            "physical_or_electrical_acceptance": False}
        return results

    def render_candidate(self, contexts, waist, candidate):
        for fid, ctx in contexts.items():
            for tag, q in (("endpoint", candidate["path"]["outward_arm_rad"][0]),
                           ("home", candidate["path"]["home_arm_rad"])):
                ctx.robot.set(self.cr.WAIST, waist); ctx.robot.set(self.cr.LEFT, q); ctx.robot.fk()
                for gid in range(ctx.robot.m.ngeom):
                    if ctx.robot.m.geom(gid).name.startswith("tool_disc"):
                        ctx.robot.m.geom_rgba[gid, 3] = .8
                        ctx.robot.m.geom_group[gid] = 0
                stage = (f"{self.args.endpoint_retract_mm:g} mm noncontact endpoint"
                         if tag == "endpoint" and self.args.endpoint_retract_mm else tag)
                self.cr.shot(ctx, self.args.output / (fid + "_" + tag + ".png"), fid + " " + stage + ", fixed waist",
                             ["Actual Rz(-90), model-specific motor placement", "Offline geometric study"])

    def run(self):
        a = self.args
        if a.verify:
            candidate = json.loads(a.verify.read_text())["candidate"]
            flange = np.array(candidate["common_nominal_flange_pose"])
            boxes = {f: np.array(t) for f, t in candidate["world_T_boxes"].items()}
            camera = np.array(candidate["world_T_camera"])
            if a.contact_offsets or a.target_corrections:
                boxes = self.design_boxes(flange)
                camera = self.camera_pose(boxes)
                candidate["world_T_boxes"] = boxes
                candidate["world_T_camera"] = camera
                candidate["contact_offsets_mm"] = self.contact_offsets
                candidate["target_corrections"] = self.target_corrections
            else:
                self.contact_offsets.update(candidate.get("contact_offsets_mm", {}))
                self.target_corrections.update(candidate.get("target_corrections", {}))
            waist = np.radians(candidate["waist_model_deg"])
            previous_retract = float(candidate.get("endpoint_retract_mm", 0))
            if a.endpoint_retract_mm != previous_retract:
                if a.endpoint_retract_mm < previous_retract:
                    raise ValueError("Cannot extend a cached route deeper without solving it")
                endpoint = flange.copy()
                endpoint[:3, 3] -= endpoint[:3, 2] * a.endpoint_retract_mm / 1000
                poses = np.array(candidate["path"]["outward_flange_poses"])
                distances = np.linalg.norm(poses[:, :3, 3] - endpoint[:3, 3], axis=1)
                cut = int(np.argmin(distances))
                if distances[cut] > 1e-8:
                    raise ValueError("Cached route has no exact waypoint at requested standoff")
                for key in ("outward_arm_rad", "outward_waist_arm_rad", "outward_flange_poses"):
                    candidate["path"][key] = candidate["path"][key][cut:]
                candidate["path"]["rise_mm"] -= a.endpoint_retract_mm - previous_retract
                candidate["endpoint_retract_mm"] = a.endpoint_retract_mm
                candidate["common_endpoint_flange_pose"] = endpoint
                candidate["final_arm_rad"] = candidate["path"]["outward_arm_rad"][0]
            contexts = {f: self.build(f, boxes, camera, a.output / "verify" / f) for f in FIXTURES}
            validation = self.strict_validate(contexts, waist, candidate["path"])
            candidate["strict_validation"] = validation
            candidate["loaded_geometry_mating_metrics"] = self.mating_metrics(contexts, waist, candidate["path"]["outward_arm_rad"][0])
            if a.visibility_check:
                candidate["home_camera_visibility"] = self.home_camera_visibility(contexts, waist, candidate)
            if a.render:
                self.render_candidate(contexts, waist, candidate)
            self.finish(candidate)
            return
        # A level support is a design variable; do not inherit the old one-degree tilt.
        waists = a.waist_deg or [[-33.78, 35.11, 6.84, -46.09], [-25.5, 15.9, 35.56, -122.92],
                                [-25, 20, 10, -80], [-25, 20, 10, -100], [-20, 0, 20, -90],
                                [-15, 5, 5, -60], [-40, 35, 5, -90], [-10, 10, 20, -120]]
        positions = a.positions or [[.42, .04, .94], [.42, .04, .89], [.45, .10, .90], [.50, .12, .95],
                                   [.40, .15, .90], [.50, .0, .90], [.35, .15, .90], [.45, .2, .95],
                                   [.40, .04, .84], [.50, .10, 1.0], [.55, .10, .85]]
        candidates = [(p, yaw, w) for yaw in a.yaws_deg for p in positions for w in waists]
        if self.design:
            pose = np.asarray(self.design["common_nominal_flange_pose"], float)
            candidates = [(pose[:3, 3].tolist(), math.degrees(math.atan2(pose[1, 0], pose[0, 0])),
                           self.design["waist_model_deg"])]
        best = None
        reference_flange = transform(positions[0], a.yaws_deg[0])
        refboxes = self.design_boxes(reference_flange)
        reference = self.build("JM160", refboxes, self.camera_pose(refboxes), a.output / "reference")
        for index, (xyz, yaw, wd) in enumerate(candidates):
            if time.monotonic() >= self.deadline:
                break
            waist = np.radians(wd)
            nominal = (np.asarray(self.design["common_nominal_flange_pose"], float)
                       if self.design else transform(xyz, yaw))
            flange = nominal.copy()
            flange[:3, 3] -= flange[:3, 2] * a.endpoint_retract_mm / 1000
            qfinal, qual = self.ik(reference, waist, flange, self.seeds[:12])
            if qfinal is None:
                qfinal, qual = self.ik(reference, waist, flange, self.seeds[:4], low_elbow=True)
            if qfinal is None:
                self.log.append({"index": index, "reason": "no structural final IK"}); continue
            boxes = self.design_boxes(nominal)
            camera = self.camera_pose(boxes)
            contexts = {f: self.build(f, boxes, camera, a.output / "current" / f) for f in FIXTURES}
            finals = {f: self.quality(c, waist, qfinal, flange, contact=True) for f, c in contexts.items()}
            if any(q["clear"] < 0 or not q["structural_ok"] for q in finals.values()):
                self.log.append({"index": index, "reason": "final scene", "quality": finals}); continue
            path, attempts = self.path_candidate(contexts, waist, flange, qfinal)
            row = {"index": index, "xyz": xyz, "yaw_deg": yaw, "waist_model_deg": wd,
                   "final_quality": finals, "outward_trace_passed": path is not None, "attempts": attempts}
            self.log.append(row)
            print("CANDIDATE", index, xyz, yaw, wd, bool(path), "elapsed", round(time.monotonic() - self.started, 2), flush=True)
            save(a.output / "progress.json", self.log)
            self.seeds = [qfinal] + self.seeds[:15]
            if not path:
                continue
            candidate = {"common_nominal_flange_pose": nominal, "common_endpoint_flange_pose": flange,
                         "endpoint_retract_mm": a.endpoint_retract_mm, "world_T_boxes": boxes, "world_T_camera": camera,
                         "contact_offsets_mm": self.contact_offsets,
                         "target_corrections": self.target_corrections,
                         "waist_model_deg": wd, "final_arm_rad": qfinal, "final_quality": finals, "path": path,
                         "strict_validation": self.strict_validate(contexts, waist, path),
                         "fixture_placement_mode": "one motor loaded at a time, separate indexed box poses"}
            if a.visibility_check:
                candidate["home_camera_visibility"] = self.home_camera_visibility(contexts, waist, candidate)
            candidate["loaded_geometry_mating_metrics"] = self.mating_metrics(contexts, waist, path["outward_arm_rad"][0])
            best = candidate
            save(a.output / ("candidate_%04d.json" % index), {"candidate": candidate})
            if a.render:
                self.render_candidate(contexts, waist, candidate)
            if candidate["strict_validation"]["passed"] or not self.validator:
                break
        self.finish(best)

    def finish(self, candidate):
        files = [Path(__file__), self.args.bundle, self.args.scene]
        for mod in (self.sc, self.fx, self.cr, self.sw, self.plan):
            files.append(Path(mod.__file__))
        if self.args.geometry_module:
            files.append(self.args.geometry_module)
        if self.args.validation_module:
            files.append(self.args.validation_module)
        if self.args.tool_geometry_module:
            files.append(self.args.tool_geometry_module)
        if self.args.contact_offsets:
            files.append(self.args.contact_offsets)
        if self.args.target_corrections:
            files.append(self.args.target_corrections)
        if self.args.fixture_revision:
            files.append(self.args.fixture_revision)
        if self.args.design_config:
            files.append(self.args.design_config)
        if self.args.qr_layout:
            files.append(self.args.qr_layout)
        doc = {"candidate": candidate, "attempts": self.log, "elapsed_s": time.monotonic() - self.started,
               "full_route_verified": False,
               "full_nominal_route_verified": bool(candidate and candidate["strict_validation"]["passed"] and self.args.endpoint_retract_mm == 0),
               "full_noncontact_cycle_verified": bool(candidate and candidate["strict_validation"]["passed"] and self.args.endpoint_retract_mm > 0),
               "requested_path_verified": bool(candidate and candidate["strict_validation"]["passed"]),
               "mating_verified": False,
               "controller_writes": False, "electrical_test_verified": False, "physical_execution_verified": False,
               "actual_flange_T_tcp": self.mount, "geometry_module": str(self.args.geometry_module),
               "endpoint_retract_mm": self.args.endpoint_retract_mm,
               "endpoint_is_noncontact_standoff": self.args.endpoint_retract_mm > 0,
               "contact_offsets_mm": self.contact_offsets,
               "target_corrections": self.target_corrections,
               "proposed_fixture_revision": self.fixture_revision,
               "bounded_contact_horizon": self.args.bounded_contact_horizon,
               "contact_horizon_coverage": self.horizon_proofs,
               "assumptions": ["One motor loaded at a time", "Distinct indexed motor poses per model", "Level motor supports",
                               "Fixed waist throughout each cycle", "No bracket or flange rotation change",
                               "Nominal mating pose is not independently confirmed contact depth"],
               "sources_sha256_at_load": self.loaded_source_hashes,
               "sources_sha256_at_finish": {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
               "geometry_sha256_at_build": self.geometry_hashes}
        save(self.args.output / "result.json", doc)
        print("FINAL", bool(candidate), doc["full_route_verified"], doc["elapsed_s"], flush=True)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("scene", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sim-root", type=Path, required=True)
    p.add_argument("--control-root", type=Path, required=True)
    p.add_argument("--vision-root", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--seed-results", type=Path, nargs="*", default=[])
    p.add_argument("--geometry-module", type=Path)
    p.add_argument("--geometry-fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    p.add_argument("--validation-module", type=Path)
    p.add_argument("--tool-geometry-module", type=Path)
    p.add_argument("--bounded-contact-horizon", action="store_true",
                   help="Audit and reduce near-contact reporting without changing collision rules")
    p.add_argument("--verify", type=Path)
    p.add_argument("--design-config", type=Path,
                   help="Small design JSON with fixed waist, flange/camera poses, route settings and one IK seed")
    p.add_argument("--positions", type=float, nargs=3, action="append")
    p.add_argument("--waist-deg", type=float, nargs=4, action="append")
    p.add_argument("--yaws-deg", type=float, nargs="+", default=[60, 0, 120, -60, 180, -120])
    p.add_argument("--camera-height-mm", type=float, default=550)
    p.add_argument("--endpoint-retract-mm", type=float, default=0,
                   help="Positive: validate a noncontact endpoint this far before nominal mating")
    p.add_argument("--contact-offsets", type=Path,
                   help="JSON model -> signed insertion offset mm relative to saved nominal target")
    p.add_argument("--target-corrections", type=Path,
                   help="JSON model -> {yaw_deg, translation_mm}: nominal box coordinates to measured box coordinates")
    p.add_argument("--fixture-revision", type=Path,
                   help="Explicit design-only tool revision JSON consumed by the CAD tool builder")
    p.add_argument("--rises-mm", type=float, nargs="+", default=[60, 100, 40, 150])
    p.add_argument("--escape-azimuths-deg", type=float, nargs="+", default=[90, 120, 60, 180, 0, -90])
    p.add_argument("--escape-mm", type=float, default=280)
    p.add_argument("--render", action="store_true")
    p.add_argument("--visibility-check", action="store_true")
    p.add_argument("--qr-layout", type=Path,
                   help="Camera screen JSON containing per-model proposed_QR_patch bounds at box z=0")
    return p


def study_from_arguments(argv=None):
    args = argument_parser().parse_args(argv)
    if args.design_config:
        design = json.loads(args.design_config.read_text())
        for key in ("endpoint_retract_mm", "rises_mm", "escape_azimuths_deg", "escape_mm"):
            if key in design:
                setattr(args, key, design[key])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    return Study(args)


def main():
    study_from_arguments().run()


if __name__ == "__main__":
    main()
