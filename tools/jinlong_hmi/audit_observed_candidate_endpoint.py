#!/usr/bin/env python3
"""Read a saved structural endpoint and audit it in its own observed scene.

The search's structural screening uses a coarse JM160 reference. Its stored
collision pair is not a diagnosis for another motor. This command reconstructs
the selected model and reports raw MuJoCo contacts without running IK or motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from render_shared_station_video import study_args
from solve_shared_motor_station import SharedStationStudy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output = args.output.resolve()
    if (args.output / 'endpoint_audit.json').exists():
        raise ValueError('Audit already exists; choose a fresh output directory')
    result = json.loads(args.result.read_text())
    candidate = json.loads(args.candidate.read_text())['candidate']
    values = study_args(result, args.output)
    study = SharedStationStudy(values)
    study.station = np.asarray(candidate['world_T_station'])
    boxes = {fid: np.asarray(T) for fid, T in candidate['world_T_boxes'].items()}
    ctx = study.build(args.model, boxes, np.asarray(candidate['world_T_camera']), args.output / 'scene')
    saved = candidate['solutions'][args.model]['endpoint']
    q = np.asarray(saved['left_rad'])
    waist = np.radians(candidate['waist_model_deg'])
    target = np.asarray(candidate['solutions'][args.model]['endpoint_flange_pose'])
    quality = study.quality(ctx, waist, q, target)
    robot = ctx.robot
    box_geometry = json.loads(Path(ctx.cfg['model_specific_box_collision']).read_text())
    contacts, grouped = [], {}
    box_prefix = f'box_{args.model}_cad_'
    for index in range(robot.d.ncon):
        contact = robot.d.contact[index]
        names = [robot._gname(contact.geom1), robot._gname(contact.geom2)]
        item = {'pair': names, 'raw_distance_mm': float(contact.dist * 1000),
                'body_padding_mm': float(robot.body_pad(contact) * 1000),
                'effective_clearance_mm': float((contact.dist - robot.body_pad(contact)) * 1000),
                'skipped': bool(robot.skip_contact(contact)),
                'contact_position_world_m': contact.pos.tolist()}
        for name in names:
            if name.startswith(box_prefix):
                part_index = int(name.removeprefix(box_prefix))
                part = box_geometry['parts'][part_index]
                item['observed_surface_part'] = {
                    'index': part_index, 'source_box_index': part.get('source_box_index'),
                    'unknown_depth': part.get('unknown'), 'source_box': part.get('source_box'),
                    'interpretation': ('Missing-depth column conservatively occupied from the rim'
                                       if part.get('unknown') else 'Observed source surface column/solid'),
                }
        contacts.append(item)
        key = tuple(sorted(names))
        row = grouped.setdefault(key, {'pair': list(key), 'raw_contact_count': 0,
                                       'minimum': item})
        row['raw_contact_count'] += 1
        if item['effective_clearance_mm'] < row['minimum']['effective_clearance_mm']:
            row['minimum'] = item
    contacts.sort(key=lambda x: x['effective_clearance_mm'])
    pairs = sorted(grouped.values(), key=lambda x: x['minimum']['effective_clearance_mm'])
    box = boxes[args.model]
    socket_geometry = []
    for index in range(1, 4):
        name = f'tool_sock_{args.model}_{index}'
        geom = robot.m.geom(name)
        axis = robot.d.geom_xmat[geom.id].reshape(3, 3)[:, 2]
        center = robot.d.geom_xpos[geom.id]
        endpoints = np.array([center - axis * geom.size[1], center + axis * geom.size[1]])
        local = (endpoints - box[:3, 3]) @ box[:3, :3]
        axial_projection = float(axis @ box[:3, 2])
        radial_z = float(geom.size[0] * np.sqrt(max(0, 1 - axial_projection**2)))
        socket_geometry.append({'name': name, 'endpoints_world_mm': (endpoints * 1000).tolist(),
                                'endpoints_observed_mm': (local * 1000).tolist(),
                                'radius_mm': float(geom.size[0] * 1000),
                                'lowest_surface_above_rim_mm': float((-local[:, 2].max() - radial_z) * 1000)})
    warnings = {str(warning): int(robot.d.warning[warning].number)
                for warning in range(len(robot.d.warning)) if robot.d.warning[warning].number}
    report = {
        'status': 'actual_model_endpoint_collision_audit_only', 'model': args.model,
        'source_result': str(args.result.resolve()), 'source_result_sha256': sha(args.result),
        'source_candidate': str(args.candidate.resolve()), 'source_candidate_sha256': sha(args.candidate),
        'auditor_sha256': sha(__file__), 'saved_endpoint_retract_mm': candidate['endpoint_retract_mm'],
        'waist_model_deg': candidate['waist_model_deg'], 'world_T_station': candidate['world_T_station'],
        'left_rad': q.tolist(), 'target_flange_world_T': target.tolist(),
        'actual_flange_world_T': robot.tcp_pose('FLANGE').tolist(),
        'quality_in_actual_observed_model': quality,
        'coarse_reference_quality_not_actual_collision_diagnosis': saved['quality'],
        'observed_rim_world_z_mm': float(box[2, 3] * 1000),
        'socket_geometry': socket_geometry, 'raw_contacts': contacts, 'contact_pairs': pairs,
        'mujoco_warnings': warnings, 'collision_sources_sha256': study.geometry_hashes,
        'loaded_sources_sha256': study.loaded_source_hashes,
        'no_inverse_kinematics_or_motion_performed': True,
        'limitations': [
            'This diagnoses only the saved structural joint branch, not every IK solution.',
            'The old candidate structural quality comes from a coarse reference model and is retained only to explain the difference.',
            'Missing-depth columns reaching the rim are conservative occupied space; their top is not an observed metal lip.',
            'Positive clearance below the required threshold is insufficient clearance, not geometric penetration.',
        ],
    }
    destination = args.output / 'endpoint_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'quality': quality, 'socket_geometry': socket_geometry,
                      'worst_pairs': pairs[:6], 'mujoco_warnings': warnings}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--model', choices=['JL100', 'JL132', 'JM160'], default='JL100')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
