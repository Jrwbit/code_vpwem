#!/usr/bin/env python3
"""Offline fixed motor station search; all models share one foot datum and table.

Unlike the earlier shared-flange study, box transforms are immutable intrinsic
motor geometry. Each model gets its own arm target and route, with one waist.
Nothing here imports a hardware client or writes controller commands.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import solve_fixed_waist_cycle as base
import shared_station_constraints as constraints
import shared_home_connector as connector_module

FIXTURES = base.FIXTURES


def station_transform(xyz, yaw_deg):
    T = np.eye(4)
    T[:3, :3] = base.rz(math.radians(yaw_deg))
    T[:3, 3] = xyz
    return T


def robot_installation_transform(xy_mm, fixed_camera_calibration):
    """Candidate robot installation in the original calibration lab frame."""
    xy = np.asarray(xy_mm, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('Robot installation requires two finite XY translations in mm')
    if np.any(xy != 0) and not fixed_camera_calibration:
        raise ValueError('Nonzero robot installation XY requires a fixed camera calibration')
    return station_transform([xy[0] / 1000, xy[1] / 1000, 0], 0)


def validate_camera_lift(lift_mm, fixed_camera_calibration):
    """Validate an explicit offline vertical relocation from saved calibration."""
    lift = np.asarray(lift_mm, float)
    if lift.shape != () or not np.isfinite(lift) or lift < 0:
        raise ValueError('Camera lift requires one finite nonnegative height in mm')
    if lift != 0 and not fixed_camera_calibration:
        raise ValueError('Nonzero camera lift requires a fixed camera calibration')
    return float(lift)


def provisional_geometry():
    boxes = {}
    motors = {}
    for f, height in zip(FIXTURES, (.40, .35, .25)):
        T = base.transform([0, 0, height], 0)
        boxes[f] = T.tolist()
        M = np.eye(4); M[:3, 3] = [-.07, 0, (height - .07) / 2]
        motors[f] = {"T": M.tolist(), "size": [.35, .28, height - .07]}
    return {"station_T_boxes": boxes, "motor_envelopes": motors,
            "intrinsic_geometry_confirmed": False,
            "assumptions": ["Temporary historical 400/350/250 mm box heights", "Temporary common box XY and orientation", "Motor envelopes are approximate"]}


class SharedStationStudy(base.Study):
    def __init__(self, args):
        super().__init__(args)
        self.station_geometry = (json.loads(args.station_geometry.read_text())
                                 if args.station_geometry else provisional_geometry())
        self.station_geometry_validation = constraints.validate_station_geometry(self.station_geometry)
        self.local_boxes = {f: np.asarray(self.station_geometry["station_T_boxes"][f], float) for f in FIXTURES}
        for f, T in self.local_boxes.items():
            if not np.allclose(T[:3, 2], [0, 0, -1]):
                raise ValueError(f + ': motor box must open upwards at a level foot datum')
        self.seed_cache = {f: self.seeds.copy() for f in FIXTURES}
        for path in args.arm_seed_results:
            d = json.loads(path.read_text())
            c = d.get('candidate') or d.get('chosen')
            if not c:
                continue
            if 'path' in c:
                for f in FIXTURES:
                    self.seed_cache[f].insert(0, np.asarray(c['path']['outward_arm_rad'][0]))
            for f, sol in c.get('solutions', {}).items():
                self.seed_cache[f].insert(0, np.asarray(sol.get('endpoint', sol.get('pre30', sol))['left_rad']))
            for f, path_data in c.get('paths', {}).items():
                self.seed_cache[f].insert(0, np.asarray(path_data['outward_arm_rad'][0]))
        self.station_design = json.loads(args.station_design.read_text()) if args.station_design else {}
        self.lab_T_robot = robot_installation_transform(
            getattr(args, 'robot_installation_xy_mm', [0, 0]),
            getattr(args, 'fixed_camera_calibration', None))
        self.camera_lift_design_mm = validate_camera_lift(
            getattr(args, 'camera_lift_mm', 0),
            getattr(args, 'fixed_camera_calibration', None))
        self.calibration_lab_T_camera = None
        self.lab_T_camera = None
        self.fixed_world_T_camera = None
        if getattr(args, 'fixed_camera_calibration', None):
            self.load_fixed_camera_calibration(args.fixed_camera_calibration)
        for f, q in self.station_design.get('endpoint_seed_arm_rad', {}).items():
            self.seed_cache[f].insert(0, np.asarray(q, float))
        for key in ('endpoint_retract_mm', 'rises_mm', 'escape_azimuths_deg', 'escape_mm'):
            if key in self.station_design:
                setattr(args, key, self.station_design[key])
        self.station = np.eye(4)
        for path in (Path(__file__), Path(constraints.__file__), Path(connector_module.__file__), args.station_geometry, args.station_design, *args.arm_seed_results):
            if path:
                self.loaded_source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def load_fixed_camera_calibration(self, path):
        """Freeze a candidate installation, optionally raised in laboratory Z."""
        from helios_hmi import urdf_kin
        calibration = json.loads(path.read_text())
        if (calibration.get('base_frame') != 'rokae_left_sdk_base'
                or calibration.get('translation_unit') != 'm'):
            raise ValueError('Fixed camera requires rokae_left_sdk_base calibration in metres')
        reference_waist = np.asarray(calibration['waist_model_rad'], float)
        if reference_waist.shape != (4,) or not np.isfinite(reference_waist).all():
            raise ValueError('Fixed camera requires four finite calibration-reference waist angles')
        base_camera = constraints._transform(calibration['base_T_camera'], 'calibrated base_T_camera')
        urdf_path = urdf_kin.default_urdf_path()
        kin = urdf_kin.UrdfKinematics(urdf_path)
        calibration_lab_camera = kin.link_pose('AR5-5_07L_base', dict(zip(
            urdf_kin.WAIST_JOINTS, reference_waist))) @ base_camera
        lab_camera = calibration_lab_camera.copy()
        lab_camera[2, 3] += self.camera_lift_design_mm / 1000
        # The explicit lift is physical design movement in laboratory Z;
        # it never rotates or tracks a station/working waist. Robot installation
        # changes only the model origin. Neither operation edits calibration.
        model_camera = np.linalg.inv(self.lab_T_robot) @ lab_camera
        designed_camera = self.station_design.get('world_T_camera')
        if designed_camera is not None and not np.allclose(
                constraints._transform(designed_camera, 'design world_T_camera'),
                model_camera, atol=1e-8, rtol=0):
            raise ValueError('Station design camera conflicts with the calibrated camera and explicit lift')
        self.calibration_lab_T_camera = calibration_lab_camera
        self.lab_T_camera = lab_camera
        self.fixed_world_T_camera = model_camera
        for source in (path, Path(urdf_kin.__file__), urdf_path):
            source = Path(source).resolve()
            self.loaded_source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()

    def frame_metadata(self, station=None, boxes=None, camera=None):
        """Separate explicit physical lift from robot-coordinate origin changes."""
        if self.lab_T_camera is None:
            return None
        model_camera = self.fixed_world_T_camera if camera is None else np.asarray(camera, float)
        if not np.allclose(self.lab_T_robot @ model_camera, self.lab_T_camera,
                           atol=1e-10, rtol=0):
            raise ValueError('Candidate camera moved in the lab frame')
        return {
            'lab_frame': 'Robot base_link at the saved camera calibration installation',
            'model_frame': 'Robot base_link at the candidate XY installation',
            'legacy_world_keys_are_model_frame': True,
            'robot_installation_design_only': True,
            'robot_installation_translation_xy_mm': (self.lab_T_robot[:2, 3] * 1000).tolist(),
            'robot_installation_rotation_and_height_unchanged': True,
            'lab_T_robot': self.lab_T_robot.tolist(),
            'calibration_lab_T_camera': self.calibration_lab_T_camera.tolist(),
            'lab_T_camera': self.lab_T_camera.tolist(),
            'lab_T_station': None if station is None else (self.lab_T_robot @ np.asarray(station)).tolist(),
            'lab_T_boxes': None if boxes is None else {
                f: (self.lab_T_robot @ np.asarray(T)).tolist() for f, T in boxes.items()},
            'camera_fixed_in_lab': True,
            'camera_fixed_during_cycle': True,
            'camera_lift_design_mm': self.camera_lift_design_mm,
            'camera_unchanged_from_calibration': self.camera_lift_design_mm == 0,
            'vertical_only_design': True,
            'camera_installation_design_only': self.camera_lift_design_mm != 0,
            'physical_camera_movement_mm': self.camera_lift_design_mm,
            'physical_camera_translation_mm': [0., 0., self.camera_lift_design_mm],
            'physical_camera_rotation_deg': 0.,
            'camera_model_transform_change_is_coordinate_change_only': self.camera_lift_design_mm == 0,
            'chassis_joints_changed': False,
        }

    def camera_pose(self, boxes):
        if self.fixed_world_T_camera is not None:
            return self.fixed_world_T_camera.copy()
        if self.station_design.get('world_T_camera') is not None:
            return np.asarray(self.station_design['world_T_camera'], float)
        T = super().camera_pose(boxes)
        offset = np.asarray(self.args.camera_offset_xy_mm, float) / 1000
        if np.linalg.norm(offset) > 0:
            points = np.array([t[:3, 3] for t in boxes.values()])
            T[:2, 3] = points[:, :2].mean(0) + offset
            z = points.mean(0) - T[:3, 3]; z /= np.linalg.norm(z)
            x = np.cross(z, [0.,1.,0.]); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            T[:3, :3] = np.column_stack([x,y,z])
        return T

    def boxes_for(self, station):
        return {f: station @ self.local_boxes[f] for f in FIXTURES}

    def build(self, fid, boxes, camera, folder):
        folder.mkdir(parents=True, exist_ok=True)
        doc = copy.deepcopy(self.base)
        doc.pop('viz_boxes', None)
        # Observed box frames require the shared foot datum explicitly; a bare
        # box pose cannot distinguish the old CAD frame from the RGB-D frame.
        doc['world_T_station'] = self.station.tolist()
        old_box = np.asarray(self.base['box_T_base_link'])
        delta = boxes[fid] @ np.linalg.inv(old_box)
        table_z = float(self.station[2, 3])
        doc['obstacles'] = [o for o in doc['obstacles'] if o['name'] != 'motor_support' and not o['name'].startswith('motor_fitting_')]
        for ob in doc['obstacles']:
            if ob['name'].startswith('box_'):
                ob['T'] = (delta @ np.asarray(ob['T'])).tolist()
            elif ob['name'] == 'motor_body':
                envelope = self.station_geometry['motor_envelopes'][fid]
                ob['T'] = (self.station @ np.asarray(envelope['T'])).tolist()
                ob['size'] = envelope['size']
            elif ob['name'] == 'table':
                ob['center_xy'] = self.station[:2, 3].tolist()
                ob['z_from_floor'] = [table_z - doc['floor_z'] - .06, table_z - doc['floor_z']]
        for auxiliary in self.station_geometry.get('motor_aux_envelopes', {}).get(fid, []):
            doc['obstacles'].append({'name': auxiliary['name'],
                                     'T': (self.station @ np.asarray(auxiliary['T'])).tolist(),
                                     'size': auxiliary['size'], 'rgba': [.35,.38,.42,1]})
        doc['_note'] = 'Offline shared foot datum/table study; no model-specific motor movement. Motor and box source limitations are recorded in shared station geometry.'
        doc['box_T_base_link'] = boxes[fid].tolist()
        self.cr.apply_camera(doc, camera, center='camera', half_xy=.35, below_lens=.05)
        doc['frame_metadata'] = self.frame_metadata(self.station, boxes, camera)
        doc['terminal_fixture'] = fid
        doc['left_arm_soft_limits_rad'] = {'lower': [self.limits[n][0] for n in self.sc.LEFT],
                                          'upper': [self.limits[n][1] for n in self.sc.LEFT]}
        doc['static_pose'] = self.data['static_pose']
        self.configure(doc, self.fx, self.site, self.args.sim_root, folder)
        target = boxes[fid] @ self.local_flange_target(fid)
        target[:3, 3] -= target[:3, 2] * self.args.endpoint_retract_mm / 1000
        doc['targets'] = [{'id': fid, 'tcp': 'FLANGE', 'T': target.tolist()}]
        use_geometry = self.geometry is not None and fid in self.args.geometry_fixtures
        if use_geometry:
            self.geometry.modify_scene(doc, fid)
        if self.tool_geometry:
            if not use_geometry:
                raise ValueError('Precise tool CAD requires the model-specific box builder')
            self.tool_geometry.modify_scene(doc, self.mount)
        for key in ('model_specific_box_collision', 'tool_cad_collision', 'tool_mesh'):
            if doc.get(key):
                path = Path(doc[key])
                self.geometry_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        doc['shared_motor_station'] = {'world_T_station': self.station.tolist(),
                                      'station_T_box': self.local_boxes[fid].tolist(),
                                      'tabletop_world_z_m': table_z,
                                      'no_model_specific_placement_or_height_adjustment': True}
        doc['shared_motor_station_validation'] = constraints.validate_station_scene(
            self.station, self.station_geometry, boxes, doc, world_T_camera=camera)
        def build_model(cfg):
            m = self.geometry.build_model(cfg, fid, self.original_build) if use_geometry else self.original_build(cfg)
            m = self.select(self.apply_model(m, self.site), fid)
            if self.args.bounded_contact_horizon:
                self.horizon_proofs[fid] = base.bounded_contact_horizon(m)
            return m
        self.cr.build = build_model
        ctx = self.cr.Ctx(copy.deepcopy(doc), viz=False)
        ctx.fixture = fid
        ctx.robot.active_motor = fid
        for n in self.cr.WAIST + self.cr.LEFT:
            a, b = ctx.robot.lim[n]; c, d = self.limits[n]
            ctx.robot.lim[n] = (max(a, c), min(b, d))
        base.save(folder / 'scene_used.json', doc)
        return ctx

    def find_ik(self, ctx, fid, waist, target, previous=(), collision=False):
        robot = ctx.robot
        lo, hi = self.cr._bounds(robot, self.cr.LEFT, .05)
        seeds = list(previous) + self.seed_cache[fid][:8] + list(self.rng.uniform(lo, hi, (self.args.random_seeds, 7)))
        robot.set(self.cr.WAIST, waist)
        best = None
        for solver in (self.sw.arm_ik, self.sw.ik_low_elbow):
            for seed in seeds:
                if time.monotonic() > self.deadline:
                    return best
                ok, q = solver(robot, target, 'FLANGE', seed)
                if not ok:
                    continue
                qual = self.quality(ctx, waist, q, target)
                if qual['structural_ok'] and (not collision or qual['clear'] >= .008):
                    self.seed_cache[fid] = [q.copy()] + self.seed_cache[fid][:7]
                    return {'left_rad': q, 'quality': qual}
        return best

    def one_path(self, ctx, fid, waist, endpoint, qfinal):
        failures = []
        route_design = self.station_design.get('model_route_parameters', {}).get(fid, {})
        rises = [route_design['rise_mm']] if 'rise_mm' in route_design else self.args.rises_mm
        azimuths = [route_design['escape_azimuth_deg']] if 'escape_azimuth_deg' in route_design else self.args.escape_azimuths_deg
        fixed_distance = route_design.get('escape_mm')
        for rise in rises:
            q = np.asarray(qfinal)
            rows = [q.copy()]; poses = [endpoint.copy()]
            failure = None
            for h in np.arange(.005, rise / 1000 + 1e-6, .005):
                T = endpoint.copy(); T[2, 3] += h
                next_q, qual = self.ik(ctx, waist, T, [q])
                if next_q is None or np.max(np.abs(next_q-q)) > math.radians(12) or qual['clear'] < .008:
                    failure = {'stage': 'vertical', 'rise_mm': h*1000, 'quality': qual}; break
                rows.append(next_q); poses.append(T); q = next_q
            if failure:
                failures.append(failure); continue
            for azimuth in azimuths:
                qr = q.copy(); rr = rows.copy(); pp = poses.copy()
                first_adjust_valid_distance = None
                direction = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth)), 0])
                max_distance_mm = fixed_distance if fixed_distance is not None else self.args.escape_mm + 250
                for distance in np.arange(.01, max_distance_mm/1000 + 1e-6, .01):
                    T = poses[-1].copy(); T[:3, 3] += distance * direction
                    qnext, qual = self.ik(ctx, waist, T, [qr])
                    if qnext is None or np.max(np.abs(qnext-qr)) > math.radians(12) or qual['clear'] < .008:
                        failure = {'stage': 'lateral', 'azimuth_deg': azimuth, 'distance_mm': distance*1000, 'quality': qual}; break
                    rr.append(qnext); pp.append(T); qr = qnext
                    # Continue the same Cartesian ray until the complete hand is
                    # outside the adjustment exclusion zones; swapping tool yaw
                    # is not permitted at a merely collision-free box hover.
                    minimum_distance = fixed_distance if fixed_distance is not None else self.args.escape_mm
                    if distance < minimum_distance / 1000 - 1e-8:
                        continue
                    if self.cr.adjust_violation(ctx.robot, ctx.zones, ctx.geoms, ctx.tcp):
                        first_adjust_valid_distance = None
                        continue
                    if first_adjust_valid_distance is None:
                        first_adjust_valid_distance = distance
                    if fixed_distance is None and distance - first_adjust_valid_distance < .05 - 1e-8:
                        continue
                    return {'rise_mm': rise, 'escape_azimuth_deg': azimuth, 'escape_mm': distance*1000,
                            'outward_arm_rad': rr, 'outward_flange_poses': pp,
                            'outward_waist_arm_rad': [np.r_[waist, a] for a in rr],
                            'home_arm_rad': rr[-1], 'home_flange_pose': pp[-1]}, failures
                failures.append(failure or {'stage': 'escape', 'reason': 'No legal adjustment home within extension'})
        return None, failures

    def unify_homes(self, contexts, waist, paths, *, connector_seconds=25, connector_samples=128):
        trials = []
        homes = [('historical_folded', np.array([-1.432826609,-2.007128640,1.316061868,.497740107,1.189695716,.895000094,.587892953]))]
        homes += [(f, np.asarray(paths[f]['home_arm_rad'])) for f in FIXTURES]
        if self.station_design.get('common_home_arm_rad') is not None:
            homes.insert(0, ('design_home', np.asarray(self.station_design['common_home_arm_rad'])))
        for source, home in homes:
            static = {f:self.validator.check_segment(c, np.r_[waist,home], np.r_[waist,home], 'transit', f, waist=waist) for f,c in contexts.items()}
            if not all(r['ok'] for r in static.values()):
                trials.append({'home_source': source, 'static_checks': static, 'connectors': {}})
                continue
            visibility = None
            if getattr(self.args, 'require_camera_visible_home', False):
                first = next(iter(contexts.values()))
                visibility = self.home_camera_visibility(contexts, waist, {
                    'world_T_camera': first.cfg['camera_T_base_link'],
                    'world_T_boxes': {f: c.cfg['box_T_base_link'] for f, c in contexts.items()},
                    'path': {'home_arm_rad': home}})
                if not visibility['all_sample_rays_clear']:
                    trials.append({'home_source': source, 'static_checks': static,
                                   'camera_visibility': visibility, 'connectors': {}})
                    continue
            connectors = {}; connector_rows = {}
            print('COMMON_HOME_TRY', source, flush=True)
            for fid, ctx in contexts.items():
                q = np.asarray(paths[fid]['home_arm_rad'])
                check = self.validator.check_segment(ctx, np.r_[waist, home], np.r_[waist, q],
                                                     'transit', fid, waist=waist)
                connector = [np.r_[waist, home], np.r_[waist, q]]
                if not check['ok']:
                    seed_rows = [p['home_arm_rad'] for p in paths.values()]
                    cached = self.station_design.get('common_home_connectors', {}).get(fid)
                    if cached:
                        cached = np.asarray(cached)
                        seed_rows.extend(cached[:, 4:])
                        if np.allclose(cached[0],connector[0],atol=1e-8) and np.allclose(cached[-1],connector[-1],atol=1e-8):
                            cached_checks = [self.validator.check_segment(ctx,aa,bb,'transit',fid,waist=waist) for aa,bb in zip(cached,cached[1:])]
                            if all(e['ok'] for e in cached_checks):
                                connector = cached
                                check = {'ok': True, 'edges': cached_checks, 'reused_and_revalidated': True}
                    if not check['ok']:
                        connected = connector_module.directed_connect(ctx, connector[0], connector[-1], self.rng,
                                                                      self.validator, seconds=connector_seconds, n_samples=connector_samples,
                                                                      seed_arm_rows=seed_rows)
                        check = connected
                        if connected['ok']:
                            connector = connected['rows']
                connectors[fid] = check
                connector_rows[fid] = connector
                if not check['ok']:
                    break
            trials.append({'home_source': source, 'connectors': connectors,
                           'camera_visibility': visibility})
            if len(connectors) != 3 or not all(c['ok'] for c in connectors.values()):
                continue
            checks = {}
            for fid, ctx in contexts.items():
                path = paths[fid]
                rows = path['outward_waist_arm_rad']; poses = path['outward_flange_poses']
                common = np.r_[waist, home]
                segments = [{'name': 'common_home_to_model_home', 'kind': 'joint', 'phase': 'transit',
                             'waypoints': connector_rows[fid]}]
                for direction in (-1, 1):
                    indices = range(len(rows)-2, -1, -1) if direction == -1 else range(len(rows)-1)
                    for i in indices:
                        a, b = (i+1, i) if direction == -1 else (i, i+1)
                        segments.append({'name': ('approach_' if direction == -1 else 'return_')+str(i),
                                         'kind': 'linear', 'phase': 'corridor', 'waypoints': [rows[a],rows[b]],
                                         'Ta': poses[a], 'Tb': poses[b], 'dev_mm': .5, 'angle_deg': 1.})
                segments.append({'name': 'model_home_to_common_home', 'kind': 'joint', 'phase': 'transit',
                                 'waypoints': list(reversed(connector_rows[fid]))})
                begin = time.monotonic()
                checks[fid] = self.validator.validate_cycle(ctx, segments, fid, waist)
                checks[fid]['elapsed_s'] = time.monotonic()-begin
                path['common_home_arm_rad'] = home.copy()
                path['common_home_connector_waist_arm_rad'] = connector_rows[fid]
                path['common_home_cycle_segments'] = segments
            if all(c['ok'] for c in checks.values()):
                return {'passed': True, 'common_home_arm_rad': home, 'home_source': source,
                        'fixtures': checks, 'camera_visibility': visibility, 'trials': trials}
        return {'passed': False, 'trials': trials}

    def run(self):
        a = self.args
        waists = a.waist_deg or [[-15,5,5,-60], [-33.78,35.11,6.84,-46.09], [-25,20,10,-80],
                                [-20,0,20,-90], [-25,20,10,-100], [-10,10,20,-120], [0,0,0,-90]]
        positions = a.positions or [[.43,.05,.55], [.48,.10,.50], [.40,.15,.50], [.50,0,.45],
                                   [.35,.10,.55], [.50,.15,.60], [.45,.20,.45], [.55,.05,.50]]
        if a.station_design:
            design = self.station_design
            candidates = [(np.asarray(design['world_T_station']), design['waist_model_deg'])]
        else:
            candidates = [(station_transform(xyz, yaw), wd) for yaw, xyz, wd in itertools.product(a.yaws_deg, positions, waists)]
        self.station = candidates[0][0]
        boxes = self.boxes_for(self.station)
        # A coarse reference model is sufficient for structural IK screening.
        geometry, tool_geometry = self.geometry, self.tool_geometry
        self.geometry = self.tool_geometry = None
        reference = self.build('JM160', boxes, self.camera_pose(boxes), a.output / 'reference')
        self.geometry, self.tool_geometry = geometry, tool_geometry
        chosen = None
        for index, (station, wd) in enumerate(candidates):
            if time.monotonic() >= self.deadline:
                break
            self.station = station
            waist = np.radians(wd)
            boxes = self.boxes_for(station)
            # The box must remain on the side of the motor farther from the robot.
            far_side = {}
            for fid in FIXTURES:
                motor = station @ np.asarray(self.station_geometry['motor_envelopes'][fid]['T'])
                center = station[:2, 3] if self.station_geometry.get('model_datums') else motor[:2, 3]
                radial = center.copy()
                radial /= max(np.linalg.norm(radial), 1e-12)
                far_side[fid] = float((boxes[fid][:2, 3] - center) @ radial)
            if any(x < -1e-6 for x in far_side.values()):
                continue
            solutions = {}
            for fid in ('JM160', 'JL100', 'JL132'):
                nominal = boxes[fid] @ self.local_flange_target(fid)
                endpoint = nominal.copy(); endpoint[2, 3] += a.endpoint_retract_mm / 1000
                sol = self.find_ik(reference, fid, waist, endpoint)
                if not sol:
                    break
                high = endpoint.copy(); high[2, 3] += .03
                pre = self.find_ik(reference, fid, waist, high, [sol['left_rad']])
                if not pre:
                    break
                solutions[fid] = {'endpoint': sol, 'higher30': pre, 'nominal_flange_pose': nominal, 'endpoint_flange_pose': endpoint}
            row = {'index': index, 'world_T_station': station, 'waist_model_deg': wd, 'structural_models_passed': list(solutions)}
            self.log.append(row)
            if len(solutions) != 3:
                if index % 10 == 0:
                    print('SCREEN', index, list(solutions), 'elapsed', round(time.monotonic()-self.started, 2), flush=True)
                continue
            camera = self.camera_pose(boxes)
            candidate = {'world_T_station': station, 'station_T_boxes': self.local_boxes, 'world_T_boxes': boxes,
                         'world_T_camera': camera, 'waist_model_deg': wd, 'solutions': solutions,
                         'frame_metadata': self.frame_metadata(station, boxes, camera),
                         'endpoint_retract_mm': a.endpoint_retract_mm, 'box_far_side_offset_m': far_side, 'fixed_station_intrinsic_geometry': self.station_geometry,
                         'fixture_placement_mode': 'same foot datum, direction and table; one motor loaded at a time'}
            print('STRUCTURAL_CANDIDATE', index, 'elapsed', round(time.monotonic()-self.started,2), flush=True)
            base.save(a.output / ('structural_%04d.json' % index), {'candidate': candidate})
            if a.ik_only:
                chosen = candidate; break
            contexts = {f: self.build(f, boxes, camera, a.output / 'current' / f) for f in FIXTURES}
            candidate['shared_station_validation'] = constraints.validate_shared_station_scenes(
                self.station, self.station_geometry, boxes, {f:c.cfg for f,c in contexts.items()})
            paths = {}; checks = {}; route_failures = {}
            for fid, ctx in contexts.items():
                endpoint = solutions[fid]['endpoint_flange_pose']
                sol = self.find_ik(ctx, fid, waist, endpoint, [solutions[fid]['endpoint']['left_rad']], collision=True)
                if not sol:
                    route_failures[fid] = 'No collision-free endpoint'; break
                path, failures = self.one_path(ctx, fid, waist, endpoint, sol['left_rad'])
                route_failures[fid] = failures
                if not path:
                    break
                check = self.strict_validate({fid: ctx}, waist, path)
                paths[fid] = path; checks[fid] = check
                if not check['passed']:
                    break
            candidate['paths'] = paths
            candidate['strict_validation'] = checks
            candidate['route_failures'] = route_failures
            candidate['full_noncontact_cycles_verified'] = len(checks) == 3 and all(c['passed'] for c in checks.values())
            base.save(a.output / ('cycles_%04d.json' % index), {'candidate': candidate})
            if candidate['full_noncontact_cycles_verified']:
                candidate['common_home_validation'] = self.unify_homes(contexts, waist, paths)
                candidate['full_common_home_cycles_verified'] = candidate['common_home_validation']['passed']
            base.save(a.output / ('candidate_%04d.json' % index), {'candidate': candidate})
            row['cycle_models_passed'] = [f for f, c in checks.items() if c['passed']]
            print('CYCLES', index, row['cycle_models_passed'], flush=True)
            base.save(a.output / 'progress.json', self.log)
            old_score = len(chosen.get('strict_validation', {})) if chosen else -1
            if len(checks) >= old_score or candidate.get('full_common_home_cycles_verified'):
                chosen = candidate
            if candidate.get('full_common_home_cycles_verified'):
                break
        result = {'candidate': chosen, 'attempts': self.log, 'elapsed_s': time.monotonic()-self.started,
                  'frame_metadata': chosen.get('frame_metadata') if chosen else self.frame_metadata(),
                  'station_geometry_confirmed': self.station_geometry.get('intrinsic_geometry_confirmed', False),
                  'station_geometry_scope': self.station_geometry.get('confirmation_scope'),
                  'physical_motor_registration_confirmed': bool(
                      self.station_geometry.get('model_datums') and all(
                          d.get('physical_motor_to_box_registration_confirmed') is True
                          for d in self.station_geometry['model_datums'].values())),
                  'full_noncontact_cycles_verified': bool(chosen and chosen.get('full_noncontact_cycles_verified')),
                  'mating_verified': False, 'electrical_test_verified': False, 'physical_execution_verified': False,
                  'controller_writes': False,
                  'model_switch_path_verified': bool(chosen and chosen.get('full_common_home_cycles_verified')),
                  'full_common_home_cycles_verified': bool(chosen and chosen.get('full_common_home_cycles_verified')),
                  'sources_sha256_at_load': self.loaded_source_hashes,
                  'sources_sha256_at_finish': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.loaded_source_hashes},
                  'geometry_sha256_at_build': self.geometry_hashes,
                  'bounded_contact_horizon': self.args.bounded_contact_horizon,
                  'contact_horizon_coverage': self.horizon_proofs,
                  'run_arguments': json.loads(json.dumps(vars(a), default=str)),
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        base.save(a.output / 'result.json', result)
        print('FINAL', result['full_noncontact_cycles_verified'], round(result['elapsed_s'],2), flush=True)


def argument_parser():
    p = base.argument_parser()
    p.description = __doc__
    p.add_argument('--station-geometry', type=Path)
    p.add_argument('--station-design', type=Path)
    p.add_argument('--fixed-camera-calibration', type=Path,
                   help='Reference external camera installation from saved left-SDK-base calibration; unchanged unless an explicit vertical design lift is supplied')
    p.add_argument('--camera-lift-mm', type=float, default=0,
                   help='Offline camera lift in laboratory Z only; finite nonnegative mm, requires fixed camera calibration, does not edit it')
    p.add_argument('--robot-installation-xy-mm', type=float, nargs=2, default=[0, 0],
                   help='Offline robot installation XY translation in the calibration lab frame; camera stays fixed, direction/height unchanged')
    p.add_argument('--require-camera-visible-home', action='store_true',
                   help='Reject common homes that block sampled box/QR rays; image projection is checked separately')
    p.add_argument('--ik-only', action='store_true')
    p.add_argument('--camera-offset-xy-mm', type=float, nargs=2, default=[0,0])
    p.add_argument('--random-seeds', type=int, default=2)
    p.add_argument('--arm-seed-results', type=Path, nargs='*', default=[])
    p.set_defaults(endpoint_retract_mm=30, yaws_deg=[0,60,120,180,-60,-120], rises_mm=[30,60,100],
                   escape_azimuths_deg=[-90,90,180,120,-120,0], escape_mm=220)
    return p


def study_from_arguments(argv=None):
    args = argument_parser().parse_args(argv)
    if args.design_config:
        raise ValueError('The earlier shared-flange design format is prohibited; use --station-design')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    return SharedStationStudy(args)


def main():
    study_from_arguments().run()

if __name__ == '__main__':
    main()
