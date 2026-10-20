#!/usr/bin/env python3
"""Audit archived joint/Cartesian feedback against both controller DH tables.

Reads files only. Cartesian getters and measured joints need not describe the
same internal controller state; compare commanded joints as a separate channel.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from independent_route_audit import error, flange


def summarize(values):
    return {key: {'min': float(min(v[key] for v in values)),
                  'median': float(np.median([v[key] for v in values])),
                  'max': float(max(v[key] for v in values))}
            for key in ('position_mm', 'rotation_deg')}


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def audit(config, trace, plan_names=None):
    state = config['state']
    model = state['raw']['left_arm']['kinematic_config']
    rows = {name: np.array(model[name]['alpha_a_d_theta_deg_mm']).reshape(8, 4)
            for name in ('nominal_dh', 'effective_dh')}
    groups = {}
    observed_operations = set()
    for sample in records(trace):
        if 'error' in sample:
            continue
        if plan_names is not None and sample.get('plan', {}).get('name') not in plan_names:
            continue
        raw = sample.get('raw', {}).get('left_arm', {})
        motion = sample.get('motion', {})
        if motion.get('busy') and motion.get('operation_id'):
            observed_operations.add(motion['operation_id'])
        last = motion.get('last_result') or {}
        velocity = raw.get('qd_raw')
        if (raw.get('ok') is not True or motion.get('busy') is not False
                or last.get('ok') is not True or last.get('group') != 'left_arm'
                or last.get('operation_kind') != 'position'
                or not isinstance(velocity, list) or len(velocity) != 7
                or max(abs(v) for v in velocity) > .005):
            continue
        groups[last['operation_id']] = sample  # last settled sample per actual command
    reports = []
    for op, sample in groups.items():
        if op not in observed_operations:
            continue  # Do not attribute the pre-recording command to a new plan.
        raw = sample['raw']['left_arm']
        if raw['controller_id'] != state['raw']['left_arm']['controller_id']:
            raise ValueError('Archived controller identity mismatch')
        sdk = np.array(raw['T_base_flange_rowmajor']).reshape(4, 4)
        actual = np.array(raw['q'])
        target = np.array(sample['motion']['last_result']['target'])
        plan = sample['plan']
        reports.append({'operation_id': op, 'plan': plan['name'], 'segment': plan['segment'],
                        'step': plan['step'], 'time': sample['local_time'],
                        'joint_error_deg': np.degrees(actual-target).tolist(),
                        'effective_vs_nominal_same_actual_q': error(flange(actual, rows['effective_dh']), flange(actual, rows['nominal_dh'])),
                        'nominal_actual_vs_cartesian_getter': error(flange(actual, rows['nominal_dh']), sdk),
                        'effective_actual_vs_cartesian_getter': error(flange(actual, rows['effective_dh']), sdk),
                        'effective_command_vs_cartesian_getter': error(flange(target, rows['effective_dh']), sdk),
                        'effective_actual_vs_command': error(flange(actual, rows['effective_dh']), flange(target, rows['effective_dh']))})
    if not reports:
        raise ValueError('No settled commands found')
    official = []
    for arm in ('left_arm', 'right_arm'):
        cfg = state['raw'][arm]['kinematic_config']
        effective = np.array(cfg['effective_dh']['alpha_a_d_theta_deg_mm']).reshape(8, 4)
        for sample in cfg['controller_fk_audit']['samples']:
            official.append({'arm': arm, **error(flange(sample['q'], effective),
                                                 np.array(sample['T_base_flange']).reshape(4, 4))})
    channels = ('effective_vs_nominal_same_actual_q', 'nominal_actual_vs_cartesian_getter', 'effective_actual_vs_cartesian_getter',
                'effective_command_vs_cartesian_getter', 'effective_actual_vs_command')
    return {'controller_writes': False, 'settled_commands': len(reports),
            'official_fk_comparison': summarize(official),
            'summary': {key: summarize([r[key] for r in reports]) for key in channels},
            'commands': reports,
            'interpretation': 'Small command-vs-Cartesian residual is evidence of different feedback semantics; not independent physical accuracy or proof of controller internal implementation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-results', type=Path, help='Limit analysis to the recorded trial plan identities')
    args = p.parse_args()
    plans = None if args.run_results is None else {
        'inspection_' + trial['job'] + '.json' for trial in json.loads(args.run_results.read_text())}
    result = audit(json.loads(args.config.read_text()), args.trace, plans)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('settled_commands', 'official_fk_comparison', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
