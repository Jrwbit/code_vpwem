#!/usr/bin/env python3
"""Offline attribution of logged joint/model errors, without physical ground truth."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.control_root))
    from helios_hmi.kinematic_reference import FrameModel
    from helios_hmi.urdf_kin import UrdfKinematics, default_urdf_path
    from helios_hmi.right_display import nominal_flange

    rows = []
    with (args.trace/'joints.jsonl').open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live recorder can have an incomplete final write.
            if 'error' not in row:
                rows.append(row)
    models, groups = {}, {}
    for row in rows:
        plan_state = row.get('plan') or {}
        name = plan_state.get('name', '')
        match = re.fullmatch(r'inspection_([0-9a-f]{32})\.json', name)
        if not match:
            continue
        identifier = match[1]
        if identifier not in models:
            folder = args.trace/'artifacts/outputs/rokae_trajectory'/identifier
            if not (folder/'plan.json').exists():
                continue
            data = json.loads((folder/'input.json').read_text())
            plan = json.loads((folder/'plan.json').read_text())
            site = data['site']
            frames = FrameModel(UrdfKinematics(default_urdf_path()), data['calibration'], site)
            offset = np.asarray(site['tcp_T_objects'][site['selected_fixture']], float)
            offset[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(site['selected_fixture'], 0)/1000
            tool = np.asarray(site['flange_T_tcp'], float)
            z = np.eye(4); z[2, 3] = site.get('tcp_z_adjustment_mm', 0)/1000
            shift = np.eye(4)
            shift[:3, 3] = np.asarray(site.get('box_center_adjustments_mm', {}).get(site['selected_fixture'], [0]*3))/1000
            models[identifier] = (data, plan, frames, tool@z@offset@np.linalg.inv(shift))
        data, plan, frames, tool = models[identifier]
        index, step = plan_state.get('segment'), plan_state.get('step')
        if not isinstance(index, int) or not 0 <= index < len(plan['segments']):
            continue
        segment = plan['segments'][index]
        raw = row.get('raw') or {}
        body, arm = raw.get('body') or {}, raw.get('left_arm') or {}
        waist = np.asarray(body.get('q_raw', [])[:4], float)
        q = np.asarray(arm.get('q', []), float)
        if waist.shape != (4,) or q.shape != (7,) or not np.isfinite(np.r_[waist, q]).all():
            continue
        waist[1] += np.pi/2
        key = (identifier, index, step)
        g = groups.setdefault(key, {'job': identifier, 'segment': index, 'step': step,
                                   'name': segment['name'], 'group': segment['group'],
                                   'pause_at_target': segment.get('pause_at_target', False),
                                   'samples': 0, 'start': row['local_time'], 'waist_samples': [],
                                   'max_source_skew_s': 0, 'settled': None})
        g['samples'] += 1; g['end'] = row['local_time']; g['waist_samples'].append(waist)
        if isinstance(body.get('t'), (int, float)) and isinstance(arm.get('t'), (int, float)):
            g['max_source_skew_s'] = max(g['max_source_skew_s'], abs(body['t']-arm['t']))
        if ((row.get('motion') or {}).get('phase') == 'completed'
                and not (row.get('motion') or {}).get('busy')):
            last_result = (row.get('motion') or {}).get('last_result') or {}
            if (last_result.get('status') != 'completed' or last_result.get('group') != 'left_arm'
                    or last_result.get('safe_exit_confirmed') is not True
                    or np.asarray(last_result.get('target', [])).shape != (7,)):
                continue
            # HMI can advance its step before publishing the previous motion's
            # terminal feedback. Associate by the completed command, not merely
            # by the plan cursor; do not match a future repeated return target.
            matches = []
            for candidate in groups:
                job_id, si, ti = candidate
                if job_id != identifier or not isinstance(ti, int):
                    continue
                s = plan['segments'][si]
                if s['group'] != 'left_arm' or not 1 <= ti <= len(s['steps']):
                    continue
                if np.allclose(last_result['target'], s['steps'][ti-1]['target'], atol=1e-10, rtol=0):
                    matches.append(candidate)
            if not matches:
                continue
            selected = max(matches, key=lambda key: key[1:])
            g = groups[selected]
            segment = plan['segments'][selected[1]]
            target = np.asarray(segment['steps'][selected[2]-1]['target'])
            held = np.asarray(segment['hold']['waist'])
            actual_base, planned_base = frames.base('left_arm', waist), frames.base('left_arm', held)
            planned = planned_base@nominal_flange(target)@tool
            waist_only = actual_base@nominal_flange(target)@tool
            joint_only = planned_base@nominal_flange(q)@tool
            nominal = actual_base@nominal_flange(q)@tool
            sdk_flat = arm.get('T_base_flange_rowmajor')
            sdk = actual_base@np.asarray(sdk_flat).reshape(4, 4)@tool if sdk_flat is not None else None
            box_R = (frames.world_camera@np.asarray(data['camera_T_box']))[:3, :3]
            def delta(T):
                d = (T[:3, 3]-planned[:3, 3])*1000
                return {'norm_mm': float(np.linalg.norm(d)), 'box_axes_mm': (box_R.T@d).tolist()}
            g['settled'] = {'time': row['local_time'], 'waist_model_rad': waist.tolist(),
                            'left_rad': q.tolist(), 'joint_error_deg': np.degrees(q-target).tolist(),
                            'waist_vs_hold_deg': np.degrees(waist-held).tolist(),
                            'waist_effect': delta(waist_only), 'joint_effect': delta(joint_only),
                            'nominal_total': delta(nominal), 'sdk_total': delta(sdk) if sdk is not None else None}
    result = {'samples': len(rows), 'assumption': 'Nominal waist geometry and configured tool; model consistency is not independently measured physical error. Box axes need mapping to onsite left/right.', 'steps': []}
    for g in groups.values():
        w = np.asarray(g.pop('waist_samples'))
        g['waist_span_deg'] = np.degrees(np.ptp(w, axis=0)).tolist()
        result['steps'].append(g)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'samples': len(rows), 'steps': len(result['steps']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
