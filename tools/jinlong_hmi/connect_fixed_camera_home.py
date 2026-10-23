#!/usr/bin/env python3
"""Offline bidirectional RRT for a fixed-waist common-home connection.

Search edges use coarse joint sampling with all existing collision and H/F
adjustment rules. A reported route must pass the default dense validator on
every edge. No controller client or hardware commands are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def rrt_connect(start, goal, lower, upper, check, rng, *, seconds=180,
                step_rad=.16, seed_rows=()):
    """Search arm coordinates; ``check(a,b,strict)`` returns the full policy result."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (start.shape != goal.shape or lower.shape != start.shape or upper.shape != start.shape
            or not all(np.isfinite(q).all() for q in (start, goal, lower, upper))
            or np.any(lower >= upper) or seconds <= 0 or step_rad <= 0):
        raise ValueError('Finite matching joint vectors, ordered limits and positive search settings required')
    began = time.monotonic()
    deadline = began + seconds
    stats = {'iterations': 0, 'point_checks': 0, 'coarse_edge_checks': 0,
             'strict_edge_checks': 0, 'strict_rejected_routes': 0,
             'accepted_nodes': 0, 'restarts': 0}
    seeds = [np.asarray(q, float) for q in seed_rows]
    if any(q.shape != start.shape or not np.isfinite(q).all() for q in seeds):
        raise ValueError('Each IK seed must have the same finite joint-vector shape')
    failures = []

    def finish(ok, **extra):
        return {'ok': ok, 'elapsed_s': time.monotonic() - began, 'step_rad': step_rad,
                **stats, 'strict_failures': failures, **extra}

    for name, q in [('start', start), ('goal', goal)]:
        result = check(q, q, True)
        stats['point_checks'] += 1
        if not result['ok']:
            return finish(False, reason=name + '_is_not_a_legal_adjustment_pose', endpoint_check=result)
    direct = check(start, goal, True)
    stats['strict_edge_checks'] += 1
    if direct['ok']:
        return finish(True, arm_rows=[start, goal], checks=[direct])

    def tree(root, is_start):
        return {'q': [root.copy()], 'parents': [-1], 'is_start': is_start}

    def trace(branch, index):
        rows = []
        while index >= 0:
            rows.append(branch['q'][index])
            index = branch['parents'][index]
        return rows[::-1]

    def extend(branch, target):
        points = np.asarray(branch['q'])
        nearest = int(np.argmin(np.linalg.norm(points - target, axis=1)))
        old = points[nearest]
        delta = target - old
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return 'reached', nearest
        new = old + delta * min(1., step_rad / distance)
        result = check(new, new, False)
        stats['point_checks'] += 1
        if not result['ok']:
            return 'trapped', nearest
        result = check(old, new, False)
        stats['coarse_edge_checks'] += 1
        if not result['ok']:
            return 'trapped', nearest
        branch['q'].append(new)
        branch['parents'].append(nearest)
        stats['accepted_nodes'] += 1
        return ('reached' if distance <= step_rad else 'advanced'), len(branch['q']) - 1

    left, right = tree(start, True), tree(goal, False)
    last_progress = began
    while time.monotonic() < deadline:
        stats['iterations'] += 1
        mode = rng.random()
        if seeds and mode < .35:
            target = seeds[int(rng.integers(len(seeds)))].copy()
            if mode < .18:
                target += rng.normal(0, .12, len(start))
        elif mode < .50:
            target = right['q'][int(rng.integers(len(right['q'])))].copy()
        elif mode < .65:
            alpha = rng.random()
            target = start * (1-alpha) + goal * alpha + rng.normal(0, .5, len(start))
        else:
            target = rng.uniform(lower, upper)
        target = np.clip(target, lower, upper)
        status, i = extend(left, target)
        if status != 'trapped':
            meeting = left['q'][i]
            while time.monotonic() < deadline:
                other_status, j = extend(right, meeting)
                if other_status != 'advanced':
                    break
            else:
                break
            if other_status == 'reached':
                a, b = trace(left, i), trace(right, j)
                if not left['is_start']:
                    a, b = b, a
                rows = a + b[-2::-1]
                checks = []
                for edge, (qa, qb) in enumerate(zip(rows, rows[1:])):
                    result = check(qa, qb, True)
                    stats['strict_edge_checks'] += 1
                    checks.append(result)
                    if not result['ok']:
                        stats['strict_rejected_routes'] += 1
                        failures.append({'edge': edge, 'a': qa, 'b': qb, 'check': result})
                        break
                else:
                    return finish(True, arm_rows=rows, checks=checks)
                # Coarse feasibility is never acceptance. A dense rejection
                # discards both search trees rather than retaining a bad edge.
                left, right = tree(start, True), tree(goal, False)
                stats['restarts'] += 1
        left, right = right, left
        if time.monotonic() - last_progress >= 10:
            print('RRT_PROGRESS', stats['iterations'], len(left['q']), len(right['q']),
                  round(time.monotonic() - began, 1), flush=True)
            last_progress = time.monotonic()
    return finish(False, reason='rrt_connection_time_budget',
                  tree_sizes=[len(left['q']), len(right['q'])], last_direct_failure=direct)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'candidate', 'target_path', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--fixture', choices=['JM160', 'JL132', 'JL100'], required=True)
    p.add_argument('--common-home-fixture', default='JM160', choices=['JM160', 'JL132', 'JL100'])
    p.add_argument('--ik-seeds', type=Path)
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--step-rad', type=float, default=.16)
    p.add_argument('--seed', type=int, default=2026102313)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    from screen_shared_station_camera import rebuild
    from solve_fixed_waist_cycle import save
    r = json.loads(a.result.read_text())
    c = json.loads(a.candidate.read_text())['candidate']
    target_document = json.loads(a.target_path.read_text())
    target_path = target_document['path']
    study = rebuild(r, a.output / 'scene')
    study.station = np.asarray(c['world_T_station'])
    boxes = {f: np.asarray(t) for f, t in c['world_T_boxes'].items()}
    camera = np.asarray(c['world_T_camera'])
    waist = np.radians(c['waist_model_deg'])
    ctx = study.build(a.fixture, boxes, camera, a.output / 'scene' / a.fixture)
    start = np.asarray(c['paths'][a.common_home_fixture]['home_arm_rad'])
    goal = np.asarray(target_path['home_arm_rad'])
    lower = np.asarray([ctx.robot.lim[n][0] for n in study.cr.LEFT]) + np.radians(3)
    upper = np.asarray([ctx.robot.lim[n][1] for n in study.cr.LEFT]) - np.radians(3)
    coarse = study.validator.ClearancePolicy(sample_deg=2.)
    full = study.validator.ClearancePolicy()

    def check(qa, qb, strict):
        return study.validator.check_segment(ctx, np.r_[waist, qa], np.r_[waist, qb],
                                             'transit', a.fixture, full if strict else coarse,
                                             waist=waist)

    seeds = [start, goal]
    for trial in c.get('common_home_validation', {}).get('trials', []):
        for connector in trial.get('connectors', {}).values():
            seeds.extend(np.asarray(q)[4:] for q in connector.get('rows', []))
    if a.ik_seeds:
        seed_document = json.loads(a.ik_seeds.read_text())
        seeds.extend(np.asarray(row['arm_rad']) for row in seed_document['records'])
    seeds.extend(np.asarray(q) for q in target_path['outward_arm_rad'][::3])
    sources = [a.result, a.candidate, a.target_path, Path(__file__)]
    if a.ik_seeds:
        sources.append(a.ik_seeds)
    hashes = {str(q.resolve()): hashlib.sha256(q.read_bytes()).hexdigest() for q in sources}
    result = rrt_connect(start, goal, lower, upper, check, np.random.default_rng(a.seed),
                         seconds=a.seconds, step_rad=a.step_rad, seed_rows=seeds)
    if result['ok']:
        result['rows'] = [np.r_[waist, row] for row in result['arm_rows']]
    result.update(fixture=a.fixture, controller_writes=False,
                  world_T_camera=c['world_T_camera'], world_T_station=c['world_T_station'],
                  waist_model_deg=c['waist_model_deg'],
                  scope='One joint-space connection; not a complete three-model cycle',
                  sources_sha256_at_load=hashes,
                  sources_sha256_at_finish={q: hashlib.sha256(Path(q).read_bytes()).hexdigest() for q in hashes},
                  run_arguments=vars(a), coarse_sample_deg=2., strict_sample_deg=full.sample_deg)
    if result['sources_sha256_at_finish'] != hashes:
        raise ValueError('Source changed during RRT search')
    # Paths in arguments are metadata, not numerical arrays.
    result['run_arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    save(a.output / 'connector.json', result)
    print(json.dumps({k: result.get(k) for k in ('ok', 'reason', 'elapsed_s', 'iterations',
                                               'accepted_nodes', 'strict_rejected_routes')}), flush=True)


if __name__ == '__main__':
    main()
