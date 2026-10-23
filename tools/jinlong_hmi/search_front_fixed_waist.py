#!/usr/bin/env python3
"""Offline design search: fixed waist, translated camera/table/box, actual tool.

Outputs are research evidence, never an executable robot plan. A common designed
box pose is assumed for three interchangeable motors; none is a new measurement.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


FIXTURES = ('JM160', 'JL132', 'JL100')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('scene', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--control-root', type=Path, required=True)
    p.add_argument('--vision-root', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--positions', type=Path, help='JSON array of desired common box xyz in metres')
    p.add_argument('--waist-deg', type=float, nargs=4, action='append')
    p.add_argument('--random-seeds', type=int, default=5)
    p.add_argument('--render', action='store_true')
    p.add_argument('--initial-solution', type=Path)
    p.add_argument('--source-boxes', type=Path, help='Use separate archived world_T_boxes translated together')
    p.add_argument('--verify-result', type=Path, help='Rebuild and check a saved candidate instead of searching')
    p.add_argument('--route-search', action='store_true')
    p.add_argument('--camera-lift-mm', type=float, default=0)
    p.add_argument('--route-hovers-mm', type=float, nargs='+', default=[30])
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path[:0] = list(map(str, (args.control_root, args.sim_root, args.vision_root)))
    from sim.helios_insert import scene as sc, fixtures as fx, corridor as cr, swap as sw, plan
    from helios_hmi.controller_kinematics import apply_collision_model
    from hk_pointcloud_registration.robot.corridor_worker import configure_geometry, planning_start, planning_limits
    from hk_pointcloud_registration.robot.tilted_search import selected_terminal_model

    data = json.loads(args.bundle.read_text())
    base = json.loads(args.scene.read_text())
    site = data['site']
    if not np.allclose(site['flange_T_tcp'], [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]):
        raise ValueError('This study requires the actual Rz(-90 degree) installation')
    baseline_yaw = fx.FLANGE_T_TCP_DEG
    limits = planning_limits(data['limits'], site)
    sc.VENDOR = args.control_root / 'helios_hmi/model/Wheel_Robot'
    original_build = cr.build
    historical = json.loads((args.sim_root / 'output/20261016-helios-compact/report.json').read_text())['fixtures']
    _, home = planning_start(data)
    old_box = np.array(base['box_T_base_link'])
    old_camera = np.array(base['camera_T_base_link'])
    boxes = {f: old_box for f in FIXTURES}
    if args.source_boxes:
        boxes = {f: np.asarray(T) for f, T in json.loads(args.source_boxes.read_text())['world_T_boxes'].items()}
    rng = np.random.default_rng(20261023)
    waists = args.waist_deg or [[-33.78, 35.11, 6.84, -46.09], [-25.507413, 15.901926, 35.563485, -122.916206]]
    positions = (json.loads(args.positions.read_text()) if args.positions else
                 [[.45, .1, .95], [.5, .15, 1.0], [.4, .2, .95], [.55, .1, 1.0],
                  [.5, .0, .9], [.4, .0, .95], [.55, .25, .95], [.6, .15, 1.0],
                  [.35, .15, 1.0], [.45, .3, 1.05], [.6, .0, .9], [.45, .1, .85]])
    records, attempts = [], []
    started = time.monotonic()
    deadline = started + args.seconds
    chosen = None
    contexts = {}
    targets = {}
    common_scenes = {}
    seed_cache = {f: [np.radians(historical[f]['q_hover_deg']), np.radians(historical[f]['q_stage_deg']),
                      home, np.array(data['left'])] for f in FIXTURES}

    if args.initial_solution:
        previous = json.loads(args.initial_solution.read_text())['common']
        for fid, sol in previous['solutions'].items():
            seed_cache[fid].insert(0, np.asarray(sol['left_rad']))

    def build_context(fid, xyz, folder):
        scene = copy.deepcopy(base)
        scene.pop('viz_boxes', None)
        delta = np.asarray(xyz) - old_box[:3, 3]
        box_delta = boxes[fid] @ np.linalg.inv(old_box)
        for ob in scene['obstacles']:
            if 'T' in ob:
                T = np.array(ob['T'])
                if ob['name'].startswith('box_') or ob['name'] == 'motor_body':
                    T = box_delta @ T
                T[:3, 3] += delta; ob['T'] = T.tolist()
            elif ob['name'] == 'table':
                ob['center_xy'] = (np.array(ob['center_xy']) + delta[:2]).tolist()
                ob['z_from_floor'] = (np.array(ob['z_from_floor']) + delta[2]).tolist()
        box = boxes[fid].copy(); box[:3, 3] += delta
        scene['box_T_base_link'] = box.tolist()
        cam = old_camera.copy(); cam[:3, 3] += delta
        cam[2, 3] += args.camera_lift_mm / 1000
        cr.apply_camera(scene, cam, center='camera', half_xy=base['camera_keepout']['half_xy'][0],
                        below_lens=base['camera_keepout']['derived_from']['below_lens_m'])
        scene['terminal_fixture'] = fid
        scene['left_arm_soft_limits_rad'] = {'lower': [limits[n][0] for n in sc.LEFT],
                                            'upper': [limits[n][1] for n in sc.LEFT]}
        scene['static_pose'] = data['static_pose']
        # configure_geometry mutates this module constant but reloads original hulls.
        # Restore the source-hull angle before each independent scene construction.
        fx.FLANGE_T_TCP_DEG = baseline_yaw
        configure_geometry(scene, fx, site, args.sim_root, folder)
        offset = np.array(site['tcp_T_objects'][fid])
        offset[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(fid, 0) / 1000
        shift = np.eye(4)
        shift[:3, 3] = np.array(site.get('box_center_adjustments_mm', {}).get(fid, [0, 0, 0])) / 1000
        target = box @ shift @ np.linalg.inv(offset) @ np.linalg.inv(site['flange_T_tcp'])
        scene['targets'] = [{'id': fid, 'tcp': 'FLANGE', 'T': target.tolist()}]
        scene['_design_study'] = ('Separate archived box poses translated together, 100 estimated' if args.source_boxes else 'All motors use the same designed box opening pose, not measurements')
        cr.build = lambda cfg: selected_terminal_model(apply_collision_model(original_build(cfg), site), fid)
        ctx = cr.Ctx(copy.deepcopy(scene), viz=False)
        ctx.robot.active_motor = fid
        for n in cr.WAIST + cr.LEFT:
            a, b = ctx.robot.lim[n]; c, d = limits[n]
            ctx.robot.lim[n] = (max(a, c), min(b, d))
        save(folder / 'scene_used.json', scene)
        return ctx, target, scene

    def inspect(ctx, fid, waist, q, target, insertion):
        robot = ctx.robot
        robot.set(cr.WAIST, waist); robot.set(cr.LEFT, q); robot.fk()
        actual = robot.tcp_pose(ctx.tcp)
        dp = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        da = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
        if insertion:
            robot.enter_insertion(fid)
        try:
            quality = sw.quality(robot, ctx.tcp)
        finally:
            robot.leave_insertion()
        threshold = ctx.cfg.get('final_min_clear_m', cr.MIN_CLEAR) if insertion else cr.MIN_CLEAR
        in_limits = all(robot.lim[n][0] <= v <= robot.lim[n][1] for n, v in zip(cr.WAIST + cr.LEFT, np.r_[waist, q]))
        legal = sw.legal(quality, ctx.cfg, threshold) and dp < plan.POS_TOL and da < plan.ROT_TOL and in_limits
        return {'legal': bool(legal), 'left_rad': np.array(q).tolist(), 'left_deg': np.degrees(q).tolist(),
                'quality': quality, 'position_error_mm': dp * 1000, 'rotation_error_deg': math.degrees(da),
                'threshold_clearance_mm': threshold * 1000, 'inside_limits': bool(in_limits)}

    def solve(ctx, fid, waist, target, insertion, extra=(), random_count=None):
        robot = ctx.robot
        lo, hi = cr._bounds(robot, cr.LEFT, .05)
        seeds = list(extra) + seed_cache[fid] + list(rng.uniform(lo, hi, (args.random_seeds if random_count is None else random_count, 7)))
        begin = time.monotonic()
        local = []
        for seed in seeds:
            for solver in (sw.arm_ik, sw.ik_low_elbow, cr.ik_stage):
                if time.monotonic() >= deadline:
                    return None
                robot.set(cr.WAIST, waist)
                t0 = time.monotonic()
                ok, q = solver(robot, target, ctx.tcp, seed)
                row = inspect(ctx, fid, waist, q, target, insertion)
                attempt = {'fixture': fid, 'insertion': insertion, 'solver': solver.__name__,
                           'elapsed_s': time.monotonic() - t0, 'converged': bool(ok),
                           **{k: row[k] for k in ('legal', 'quality', 'position_error_mm', 'rotation_error_deg')}}
                attempts.append(attempt); local.append(attempt)
                if ok and row['legal']:
                    row['solve_s'] = time.monotonic() - begin
                    row['attempts'] = len(local)
                    seed_cache[fid] = [np.array(q)] + seed_cache[fid][:6]
                    return row
        return None

    def search_route(ctx, fid, waist, home, sol):
        robot = ctx.robot
        A = np.r_[waist, home]
        result = {'home_valid': False, 'passed': False, 'stage_trials': [],
                  'cartesian_deviation_acceptance_verified': False}
        route_started = time.monotonic()
        try:
            cr.check_line(ctx, A, A, adjust=True)
            result['home_valid'] = True
        except sw.PathError as e:
            result['home_failure'] = str(e)
            return result
        original_pre = np.array(sol['pre_target'])
        qpre = np.array(sol['pre30']['left_rad'])
        for hover_mm in args.route_hovers_mm:
            if hover_mm < 30:
                raise ValueError('Route hover must be at least the verified 30mm pre-position')
            prefix = [np.r_[waist, qpre]]
            pre = original_pre.copy()
            q = qpre.copy()
            vertical_ok = True
            for rise in np.linspace(0, (hover_mm - 30) / 1000, max(1, math.ceil((hover_mm - 30) / 10)) + 1)[1:]:
                target = sw.raised(original_pre, rise)
                robot.set(cr.WAIST, waist)
                ok, next_q = sw.arm_ik(robot, target, ctx.tcp, q)
                state = inspect(ctx, fid, waist, next_q, target, False)
                try:
                    if not ok or not state['legal']:
                        raise sw.PathError('Vertical retreat IK or ordinary clearance/limits')
                    cr.check_line(ctx, prefix[-1], np.r_[waist, next_q], adjust=False)
                except sw.PathError as e:
                    result['stage_trials'].append({'hover_mm': hover_mm, 'failure': str(e), 'last_state': state})
                    vertical_ok = False
                    break
                prefix.append(np.r_[waist, next_q]); q = np.array(next_q); pre = target
            if not vertical_ok:
                continue
            for angle in (0, -30, 30, -60, 60, -90, 90, 180):
                for slope in (0, .15, -.15):
                    rad = math.radians(angle)
                    u = np.array([math.sin(rad), math.cos(rad), -slope])
                    q = prefix[-1][4:].copy()
                    rows = list(prefix)
                    failure, state = '', None
                    horizontal_steps = 0
                    for step in range(1, 61):
                        if time.monotonic() - route_started >= args.seconds / 3:
                            result['budget_exhausted'] = True
                            result['elapsed_s'] = time.monotonic() - route_started
                            return result
                        target = pre.copy(); target[:3, 3] += step * .01 * u
                        robot.set(cr.WAIST, waist)
                        ok, qnext = sw.arm_ik(robot, target, ctx.tcp, q)
                        state = inspect(ctx, fid, waist, qnext, target, False)
                        if not ok or not state['legal']:
                            failure = 'IK or ordinary clearance/limits'; break
                        B = np.r_[waist, qnext]
                        try:
                            cr.check_line(ctx, rows[-1], B, adjust=False)
                        except sw.PathError as e:
                            failure = str(e); break
                        rows.append(B); q = np.array(qnext); horizontal_steps = step
                        if step < 10 or not cr.config_ok(ctx, waist, q, clear=.02, adjust=True):
                            continue
                        try:
                            link = cr.fewest_connect(ctx, A, B, rng, adjust=True, n_samples=100)
                        except sw.PathError as e:
                            failure = str(e); continue
                        outward = list(link) + list(reversed(rows[:-1]))
                        kinds = ['adjust'] * (len(link) - 1) + ['corridor'] * (len(rows) - 1)
                        checked = []
                        try:
                            for x, y, kind in zip(outward, outward[1:], kinds):
                                checked += cr.check_line(ctx, x, y, adjust=kind == 'adjust')
                                checked += cr.check_line(ctx, y, x, adjust=kind == 'adjust')
                        except sw.PathError as e:
                            failure = str(e); continue
                        return {**result, 'passed': True, 'angle_deg': angle, 'slope': slope, 'hover_mm': hover_mm,
                                'horizontal_corridor_length_mm': step * 10,
                                'minimum_clearance_mm': min(x['clear'] for x in checked) * 1000,
                                'worst': min(checked, key=lambda x: x['clear']), 'cpu_samples_both_directions': len(checked),
                                'one_way_roles': kinds, 'one_way_waypoints_research_only': outward,
                                'return_exact_reverse_checked': True, 'elapsed_s': time.monotonic() - route_started}
                    result['stage_trials'].append({'hover_mm': hover_mm, 'angle_deg': angle, 'slope': slope,
                                                   'horizontal_reached_mm': horizontal_steps * 10,
                                                   'failure': failure, 'last_state': state})
        result['elapsed_s'] = time.monotonic() - route_started
        return result

    if args.verify_result:
        chosen = json.loads(args.verify_result.read_text())['chosen']
        positions = []
        for fid in FIXTURES:
            sub = args.output / fid; sub.mkdir(exist_ok=True)
            contexts[fid], targets[fid], common_scenes[fid] = build_context(fid, chosen['box_xyz_m'], sub)

    for index, xyz in enumerate(positions):
        if time.monotonic() >= deadline:
            break
        folder = args.output / ('layout_%02d' % index); folder.mkdir(exist_ok=True)
        contexts = {}; targets = {}; common_scenes = {}
        for fid in FIXTURES:
            sub = folder / fid; sub.mkdir(exist_ok=True)
            contexts[fid], targets[fid], common_scenes[fid] = build_context(fid, xyz, sub)
        for wd in waists:
            waist = np.radians(wd)
            solutions = {}
            for fid in FIXTURES:
                ctx = contexts[fid]; target = targets[fid]
                final = solve(ctx, fid, waist, target, True)
                if final is None:
                    break
                pre_target = sw.raised(target, .03)
                pre = solve(ctx, fid, waist, pre_target, False, extra=[final['left_rad']])
                if pre is None:
                    break
                solutions[fid] = {'final': final, 'pre30': pre, 'pre_target': pre_target.tolist()}
            record = {'layout_index': index, 'box_xyz_m': xyz, 'waist_model_deg': wd, 'solutions': solutions}
            records.append(record)
            print('CANDIDATE', index, wd, 'passed', list(solutions), 'elapsed', round(time.monotonic() - started, 1), flush=True)
            save(args.output / 'search_progress.json', {'records': records, 'attempts': attempts})
            if len(solutions) == 3:
                chosen = record
                break
        if chosen:
            break

    if chosen:
        waist = np.radians(chosen['waist_model_deg'])
        if args.route_search:
            home = np.array([-1.432826609, -2.007128640, 1.316061868, .497740107, 1.189695716, .895000094, .587892953])
        chosen['home_arm_rad'] = np.array(home).tolist()
        chosen['home_is_design_pose_not_current_feedback'] = True
        chosen['routes'] = {}
        chosen['cached_candidate_valid'] = True
        for fid, sol in chosen['solutions'].items():
            ctx = contexts[fid]
            # Derived checks belong to the old scene until recomputed below.
            for key in ('final_line', 'local_ik_perturbations_only'):
                sol.pop(key, None)
            # Recompute targets from THIS scene; do not validate old q against old targets.
            sol['pre_target'] = sw.raised(targets[fid], .03).tolist()
            sol['final_recheck'] = inspect(ctx, fid, waist, sol['final']['left_rad'], targets[fid], True)
            sol['pre30_recheck'] = inspect(ctx, fid, waist, sol['pre30']['left_rad'], np.array(sol['pre_target']), False)
            sol['final'].update(sol['final_recheck'])
            sol['pre30'].update(sol['pre30_recheck'])
            valid = sol['final_recheck']['legal'] and sol['pre30_recheck']['legal']
            sol['cached_candidate_valid'] = bool(valid)
            chosen['cached_candidate_valid'] &= bool(valid)
            if not valid:
                chosen['routes'][fid] = {'passed': False, 'failure': 'Cached candidate invalid against rebuilt scene targets'}
                continue
            a, b = np.r_[waist, home], np.r_[waist, sol['pre30']['left_rad']]
            route = {'waypoints_research_only': [a.tolist(), b.tolist(), a.tolist()]}
            try:
                samples = cr.check_line(ctx, a, b, adjust=True) + cr.check_line(ctx, b, a, adjust=True)
                worst = min(samples, key=lambda x: x['clear'])
                route.update(passed=True, cpu_samples=len(samples), worst=worst)
            except sw.PathError as e:
                route.update(passed=False, failure=str(e))
            chosen['routes'][fid] = route
            if args.route_search:
                chosen['routes'][fid]['via_stage_search'] = search_route(ctx, fid, waist, home, sol)
                print('ROUTE', fid, chosen['routes'][fid]['via_stage_search']['passed'], flush=True)
                save(args.output / 'route_progress.json', chosen)
            f = np.r_[waist, sol['final']['left_rad']]
            try:
                samples = cr.check_line(ctx, b, f, adjust=False, contact_motor=fid)
                sol['final_line'] = {'passed': True, 'worst': min(samples, key=lambda x: x['clear'])}
            except sw.PathError as e:
                sol['final_line'] = {'passed': False, 'failure': str(e)}
            perturbations = []
            # Local IK sensitivity only: target displaced in the same scene.
            # No collision or placement robustness is asserted for these trials.
            for axis in range(3):
                for sign in (-1, 1):
                    target = np.array(sol['pre_target']); target[axis, 3] += sign * .005
                    t0 = time.monotonic(); ctx.robot.set(cr.WAIST, waist)
                    ok, q = sw.arm_ik(ctx.robot, target, ctx.tcp, sol['pre30']['left_rad'])
                    row = inspect(ctx, fid, waist, q, target, False)
                    perturbations.append({'axis': axis, 'offset_mm': sign * 5, 'elapsed_s': time.monotonic() - t0,
                                          'kinematic_converged': bool(ok and row['position_error_mm'] < plan.POS_TOL * 1000 and row['rotation_error_deg'] < math.degrees(plan.ROT_TOL)),
                                          'collision_verified': False, 'position_error_mm': row['position_error_mm'],
                                          'rotation_error_deg': row['rotation_error_deg']})
            sol['local_ik_perturbations_only'] = perturbations
            ctx.robot.set(cr.WAIST, waist); ctx.robot.set(cr.LEFT, sol['pre30']['left_rad']); ctx.robot.fk()
            if args.render:
                cr.shot(ctx, args.output / (fid + '_pre30.png'), fid + ' offline designed 30mm pre-position',
                        ['Actual Rz(-90) tool; common waist fixed', 'Research only; full insertion/return not accepted'])
        chosen['world_T_boxes'] = {f: s['box_T_base_link'] for f, s in common_scenes.items()}
        chosen['camera_T_base_link'] = common_scenes['JM160']['camera_T_base_link']
        chosen['camera_T_box'] = (np.linalg.inv(np.array(chosen['camera_T_base_link'])) @ np.array(common_scenes['JM160']['box_T_base_link'])).tolist()
    result = {'chosen': chosen, 'records': records, 'attempts': attempts, 'elapsed_s': time.monotonic() - started,
              'controller_writes': False, 'executable_plan_exported': False, 'full_route_verified': False,
              'assumptions': [('Archived separate boxes translated together; 100 remains estimated' if args.source_boxes else 'Common designed box pose for three motors, no new motor measurements'),
                              'One fixed camera shared by all models, translated with table and box',
                              'Original obstacle sizes and current software thresholds retained',
                              'Only local translational placement exploration; no adapter fitted'],
              'source_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.bundle, args.scene)},
              'actual_flange_T_tcp': site['flange_T_tcp'], 'baseline_source_hull_yaw_deg': baseline_yaw,
              'search_budget_s': args.seconds, 'design_camera_lift_mm': args.camera_lift_mm}
    module_sources = {'search_script': Path(__file__), 'worker': Path(sys.modules[configure_geometry.__module__].__file__),
                      'controller_kinematics': Path(sys.modules[apply_collision_model.__module__].__file__),
                      'fixtures': Path(fx.__file__), 'scene': Path(sc.__file__), 'corridor': Path(cr.__file__),
                      'swap': Path(sw.__file__), 'plan': Path(plan.__file__)}
    result['implementation_sources'] = {k: {'path': str(v.resolve()), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()}
                                        for k, v in module_sources.items()}
    result['runtime_roots'] = {k: str(getattr(args, k).resolve()) for k in ('sim_root', 'control_root', 'vision_root')}
    result['argv'] = sys.argv
    result['saved_candidate_source'] = str(args.verify_result) if args.verify_result else None
    save(args.output / 'result.json', result)
    print('FINAL', bool(chosen), result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
