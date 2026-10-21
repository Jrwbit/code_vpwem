#!/usr/bin/env python3
"""Offline endpoint IK/constraint audit using the current production model.

Runs the production scene adapter and intercepts CPU corridor search. It never
loads an HMI plan, imports an SDK, changes limits or sends controller requests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


class AuditComplete(Exception):
    pass


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--short-hover', action='store_true')
    parser.add_argument('--selected-motor-only', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    sys.path[:0] = [str(args.app), data['sim_root'], data['control_root']]
    from sim.helios_insert import corridor as cr
    from hk_pointcloud_registration.robot import corridor_worker as worker
    args.output.mkdir(parents=True, exist_ok=True)

    def audit(ctx, waist, target, seeds, rng, **kwargs):
        robot = ctx.robot
        if args.selected_motor_only:
            removed = []
            for index in range(robot.m.ngeom):
                name = robot.m.geom(index).name or ''
                if name.startswith('term_') and not name.startswith('term_'+target.id+'_'):
                    robot.m.geom_contype[index] = 0
                    robot.m.geom_conaffinity[index] = 0
                    removed.append(name)
            print('OFFLINE selected-model comparison; alternative terminals:', len(removed), flush=True)
        robot.set(cr.WAIST, waist)
        lo, hi = cr._bounds(robot, cr.LEFT, .3)
        trial_seeds = [np.asarray(s).copy() for s in seeds]
        trial_seeds += [rng.uniform(lo, hi) for _ in range(24)]
        if args.short_hover:
            saved = json.loads((args.output.parent/'endpoint-audit/endpoint_audit.json').read_text())
            trial_seeds = [np.asarray(q) for q in saved['distinct_legal_final_q']]
        report = {'controller_writes': False, 'selected_motor_only': args.selected_motor_only, 'waist_model_rad': np.asarray(waist).tolist(),
                  'seed_count': len(trial_seeds), 'poses': {}}
        from sim.helios_insert import swap as sw
        poses = [('final', target.T), ('near', target.T_pre),
                 *[(f'hover_{int(h*1000)}mm', sw.raised(target.T, h)) for h in (.04, .06, .08, .11, .15, .20)]]
        if args.short_hover:
            poses = [('free_near', target.T_pre), *[(f'hover_{int(h*1000)}mm', sw.raised(target.T, h)) for h in (.015, .02, .025, .03, .035, .04, .05, .06)]]
        valid_final = []
        for name, pose in poses:
            entries = []
            for index, seed in enumerate(trial_seeds):
                for method in (sw.arm_ik, sw.ik_low_elbow):
                    ok, q = method(robot, pose, ctx.tcp, seed)
                    if not ok:
                        continue
                    if name in ('final', 'near'):
                        robot.enter_insertion(target.id)
                    try:
                        quality = sw.quality(robot, ctx.tcp)
                    finally:
                        if name in ('final', 'near'):
                            robot.leave_insertion()
                    legal = sw.legal(quality, ctx.cfg, ctx.cfg.get('final_min_clear_m', cr.MIN_CLEAR)) if name == 'final' else sw.legal(quality, ctx.cfg)
                    entry = {'seed': index, 'method': method.__name__, 'legal': legal,
                             'q_rad': np.asarray(q).tolist(), 'quality': plain(quality)}
                    entries.append(entry)
                    if name == 'final' and legal and not any(np.max(abs(np.asarray(q)-v)) < np.radians(1) for v in valid_final):
                        valid_final.append(np.asarray(q).copy())
            report['poses'][name] = {'converged': len(entries), 'legal': sum(bool(e['legal']) for e in entries),
                                    'solutions': sorted(entries, key=lambda e: (not e['legal'], -e['quality']['clear']))[:16]}
            print(json.dumps({'pose': name, 'converged': len(entries), 'legal': report['poses'][name]['legal']}, ensure_ascii=False), flush=True)
            (args.output/'endpoint_audit.json').write_text(json.dumps(plain(report), ensure_ascii=False, indent=2))
        report['distinct_legal_final_q'] = plain(valid_final)
        (args.output/'endpoint_audit.json').write_text(json.dumps(plain(report), ensure_ascii=False, indent=2))
        raise AuditComplete()

    cr.solve_corridor = audit
    try:
        worker.run(args.input, args.output/'scene')
    except AuditComplete:
        print('Endpoint audit complete; no controller writes.', flush=True)


if __name__ == '__main__':
    main()
