#!/usr/bin/env python3
"""Compare settled SDK/joint feedback from completed inspection-return trials.

Software repeatability only: the configured kinematics are not independent
physical metrology and cannot identify absolute vision/hand-eye/TCP error.
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def mean_pose(poses):
    values = np.asarray(poses, float)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not np.isfinite(values).all():
        raise ValueError('Invalid settled SDK poses')
    result = np.eye(4)
    u, _, vt = np.linalg.svd(values[:, :3, :3].mean(axis=0))
    result[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    result[:3, 3] = values[:, :3, 3].mean(axis=0)
    return result


def pose_distance(a, b):
    delta = b[:3, 3] - a[:3, 3]
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1)
    return {'translation_mm': float(np.linalg.norm(delta)*1000),
            'rotation_deg': float(np.degrees(np.arccos(cosine)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.control_root))
    from helios_hmi.kinematic_reference import FrameModel
    from helios_hmi.urdf_kin import UrdfKinematics, default_urdf_path
    from helios_hmi.right_display import nominal_flange

    rounds = json.loads((args.run/'results.json').read_text())
    if len(rounds) != 3 or any(r['state'] != 'completed' for r in rounds):
        raise ValueError('Three completed return cycles are required')
    first = rounds[0]['job']
    folder = args.trace/'artifacts/outputs/rokae_trajectory'/first
    data = json.loads((folder/'input.json').read_text())
    plan = json.loads((folder/'plan.json').read_text())
    site = data['site']
    frames = FrameModel(UrdfKinematics(default_urdf_path()), data['calibration'], site)
    offset = np.asarray(site['tcp_T_objects'][site['selected_fixture']], float)
    offset[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(site['selected_fixture'], 0)/1000
    z = np.eye(4); z[2, 3] = site.get('tcp_z_adjustment_mm', 0)/1000
    shift = np.eye(4)
    shift[:3, 3] = np.asarray(site.get('box_center_adjustments_mm', {}).get(site['selected_fixture'], [0]*3))/1000
    tool = np.asarray(site['flange_T_tcp']) @ z @ offset @ np.linalg.inv(shift)
    box = frames.world_camera @ np.asarray(data['camera_T_box'])
    targets = {'target': plan['segments'][3]['steps'][-1]['target'],
               'home': plan['segments'][-1]['steps'][-1]['target']}
    held = np.asarray(plan['segments'][0]['hold']['waist'])
    poses, settled = {}, []
    for trial in rounds:
        for label in ('target', 'home'):
            files = sorted(args.run.glob(f'trial{trial["trial"]}_{label}_*.json'))
            if len(files) != 20:
                raise ValueError('Expected twenty settled samples per position')
            # Use the final 2.5 seconds; preserve all twenty originals separately.
            samples = [json.loads(f.read_text()) for f in files[-10:]]
            q = np.array([s['raw']['left_arm']['q'] for s in samples])
            waist = np.array([[s['joints_model_rad'][f'{i}-Joint'] for i in range(1, 5)] for s in samples])
            sdk = [np.asarray(s['raw']['left_arm']['T_base_flange_rowmajor']).reshape(4, 4) for s in samples]
            sdk_mean = mean_pose(sdk)
            world = mean_pose([frames.base('left_arm', w) @ T @ tool for w, T in zip(waist, sdk)])
            planned = frames.base('left_arm', held) @ nominal_flange(np.array(targets[label])) @ tool
            nominal = frames.base('left_arm', waist.mean(axis=0)) @ nominal_flange(q.mean(axis=0)) @ tool
            poses[trial['trial'], label] = (sdk_mean, world)
            item = {'trial': trial['trial'], 'job': trial['job'], 'position': label,
                    'left_mean_rad': q.mean(axis=0).tolist(),
                    'waist_mean_model_rad': waist.mean(axis=0).tolist(),
                    'left_target_error_deg': np.degrees(q.mean(axis=0)-targets[label]).tolist(),
                    'left_settled_span_deg': np.degrees(np.ptp(q, axis=0)).tolist(),
                    'waist_settled_span_deg': np.degrees(np.ptp(waist, axis=0)).tolist(),
                    'sdk_flange_in_arm_base': sdk_mean.tolist(),
                    'configured_fixture_in_world': world.tolist(),
                    'sdk_chain_vs_plan': pose_distance(planned, world),
                    'sdk_chain_vs_plan_box_axes_mm': (box[:3, :3].T @ (world[:3, 3]-planned[:3, 3])*1000).tolist(),
                    'nominal_joint_chain_vs_plan': pose_distance(planned, nominal)}
            settled.append(item)
    pairs = []
    for label in ('target', 'home'):
        for a, b in itertools.combinations(range(1, 4), 2):
            sa, wa = poses[a, label]; sb, wb = poses[b, label]
            pairs.append({'position': label, 'trials': [a, b],
                          'sdk_flange': pose_distance(sa, sb),
                          'configured_fixture_with_waist': pose_distance(wa, wb),
                          'fixture_delta_box_axes_mm': (box[:3, :3].T @ (wb[:3, 3]-wa[:3, 3])*1000).tolist()})
    rows = [json.loads(line) for line in (args.trace/'joints.jsonl').read_text().splitlines()]
    valid = [s for s in rows if 'error' not in s]
    trial_ids = {r['job']: r['trial'] for r in rounds}
    selected = [s for s in valid if (s.get('plan') or {}).get('name', '')[11:-5] in trial_ids]
    waist_all = np.array([[s['joints_model_rad'][f'{i}-Joint'] for i in range(1, 5)] for s in selected])
    frozen_q = np.array(targets['target'])
    fixed_waist_point = np.array([(frames.base('left_arm', w) @ nominal_flange(frozen_q) @ tool)[:3, 3] for w in waist_all])
    waist_displacement = np.linalg.norm(fixed_waist_point-fixed_waist_point[0], axis=1)*1000
    result = {'completed_rounds': 3, 'recorded_samples': len(valid), 'recording_errors': len(rows)-len(valid),
              'source_plan_sha256': data['repeatability']['original_plan_sha256'],
              'camera_T_box': data['camera_T_box'], 'world_T_box': box.tolist(),
              'waist_full_span_deg': np.degrees(np.ptp(waist_all, axis=0)).tolist(),
              'waist_change_effect_at_fixed_target_max_mm': float(waist_displacement.max()),
              'settled': settled, 'pairwise': pairs,
              'limitation': 'SDK and encoder repeatability with configured geometry; no independent physical ground truth. Vision and calibration were reused, not re-estimated.'}
    (args.output/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    columns = ['time', 'local_time', 'trial', 'segment', 'step', 'phase']
    columns += [f'{group}_{i+1}' for group, count in [('left_q_rad', 7), ('left_qd_rad_s', 7), ('waist_model_rad', 4), ('waist_controller_rad', 4), ('waist_qd_rad_s', 4), ('sdk_flange_matrix', 16)] for i in range(count)]
    with (args.output/'joints_sdk_full.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for s in valid:
            p = s.get('plan') or {}; raw = s.get('raw') or {}; arm = raw.get('left_arm') or {}; body = raw.get('body') or {}
            row = {k: s.get(k) for k in ('time', 'local_time')}
            row.update(trial=trial_ids.get(p.get('name', '')[11:-5]), segment=p.get('segment'), step=p.get('step'), phase=(s.get('motion') or {}).get('phase'))
            for group, values in [('left_q_rad', arm.get('q', [])), ('left_qd_rad_s', arm.get('qd', [])), ('waist_model_rad', [s['joints_model_rad'][f'{i}-Joint'] for i in range(1, 5)]), ('waist_controller_rad', body.get('q_raw', [])[:4]), ('waist_qd_rad_s', body.get('qd_raw', [])[:4]), ('sdk_flange_matrix', arm.get('T_base_flange_rowmajor', []))]:
                row.update({f'{group}_{i+1}': v for i, v in enumerate(values)})
            writer.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    elapsed = np.array([s['time']-selected[0]['time'] for s in selected])
    left = np.array([s['raw']['left_arm']['q'] for s in selected])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for i in range(7): axes[0].plot(elapsed, np.degrees(left[:, i]), label=f'J{i+1}')
    for i in range(4): axes[1].plot(elapsed, np.degrees(waist_all[:, i]-waist_all[0, i]), label=f'W{i+1}')
    axes[2].plot(elapsed, waist_displacement)
    for r in rounds:
        times = [s['time']-selected[0]['time'] for s in selected if (s.get('plan') or {}).get('name') == f'inspection_{r["job"]}.json']
        for ax in axes: ax.axvline(min(times), color='grey', linestyle=':', alpha=.7)
        axes[0].text(min(times), axes[0].get_ylim()[1], f'  Round {r["trial"]}', va='top')
    for ax, label in zip(axes, ['Left arm (deg)', 'Waist change (deg)', 'Waist effect at target (mm)']):
        ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0].legend(ncol=7); axes[1].legend(ncol=4); axes[-1].set_xlabel('Elapsed seconds')
    fig.tight_layout(); fig.savefig(args.output/'three_rounds_trace.png', dpi=160); plt.close(fig)
    print(json.dumps({'completed_rounds': 3, 'samples': len(valid), 'pairwise': pairs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
