#!/usr/bin/env python3
"""Assemble one reproducible, qualified offline station design from real scans.

No IK, robot, camera capture or hardware writes occur here. The source-to-
observation registrations are exact same-capture coordinate changes. The motor
foot-to-box translation remains the archived CAD nominal value and is explicitly
not physically registered. The observed orientation candidate does not convert
that nominal translation into a measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import build_observed_terminal_geometry as adapter
from shared_station_constraints import validate_station_geometry

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023'
OLD_STATION = ROOT / 'outputs/diagnostics/shared-station-20261023/geometry/shared_station_geometry.json'
OLD_SCAN160 = ROOT / 'outputs/diagnostics/full-solve-20261023/geometry/JM160_collision.json'
STATION_R_OBSERVED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
MATING_ORDER = {'JL100': ['U1', 'V1', 'W1'], 'JL132': ['W1', 'V1', 'U1'], 'JM160': ['U1', 'V1', 'W1']}
OTHER_ORDER = {'JL100': ['W2', 'U2', 'V2'], 'JL132': ['V2', 'U2', 'W2'], 'JM160': ['W2', 'U2', 'V2']}
TRANSLATION_LIMITS = [
    'Box-centre translation above the common foot datum is copied from archived whole-motor CAD, not measured from these RGB-D captures.',
    'The captures establish local box/terminal geometry and observed orientation; they do not contain independently surveyed foot datums.',
    'Box version, mounting offset and CAD-to-physical translation remain unaccepted; feasibility is an offline design candidate only.',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path, **extra):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), **extra}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def source_to_observed(template, scan, fid):
    """No fitted or guessed transform: invert two frames of the same capture."""
    adapter.require(Path(template['source_scan']).resolve() == Path(scan['source_scan']).resolve(),
                    fid + ': template and surface model are not from the same capture')
    camera_T_observed = adapter.rigid_matrix(template['camera_T_box'], fid + '.camera_T_observed')
    camera_T_source = adapter.rigid_matrix(scan['camera_T_box'], fid + '.camera_T_source')
    transform = np.linalg.inv(camera_T_observed) @ camera_T_source
    if fid != 'JM160':
        adapter.require(np.allclose(transform, np.eye(4), rtol=0, atol=1e-9),
                        fid + ': new surface model must already use the observation template frame')
        transform = np.eye(4)
    return transform


def orientation_model(document, fid):
    row = document.get('models', {}).get(fid)
    adapter.require(isinstance(row, dict), fid + ': missing observed-orientation evidence')
    rotation = np.asarray(row.get('station_R_observed'), float)
    adapter.require(rotation.shape == (3, 3) and
                    np.allclose(rotation, STATION_R_OBSERVED, rtol=0, atol=1e-9),
                    fid + ': orientation evidence differs from the explicitly proposed station orientation')
    return row


def prepare(args):
    inputs, output = args.inputs.resolve(), args.output.resolve()
    names = ('same_capture_registration_evidence.json', 'observed_geometry_manifest.json',
             'observed_station_geometry.json', 'preparation_receipt.json')
    adapter.require(not any((inputs / name).exists() for name in names),
                    'Planning inputs already exist; choose a fresh --inputs directory to preserve previous evidence')
    adapter.require(not any((output / (fid + '_collision.json')).exists() for fid in adapter.MODELS),
                    'Registered collision files already exist; choose a fresh --output directory')
    tracked = [args.station_geometry, args.orientation_evidence, args.mating_evidence,
               Path(__file__), Path(adapter.__file__)]
    station_source = json.loads(args.station_geometry.read_text())
    orientation = json.loads(args.orientation_evidence.read_text())
    validate_station_geometry(station_source)
    models, source_docs, templates = {}, {}, {}
    math_evidence = {'status': 'same_capture_coordinate_registration_only', 'models': {},
                     'physical_motor_to_foot_registration_verified': False,
                     'notes': ['These transforms register two named frames of the same archived capture.',
                               'They do not fit the scan onto whole-motor CAD or establish foot-to-box translation.']}
    for fid in adapter.MODELS:
        template_path = args.templates / (fid + '_observed_template.json')
        source_path = args.scan160 if fid == 'JM160' else args.scans / (fid + '_collision.json')
        template, source = json.loads(template_path.read_text()), json.loads(source_path.read_text())
        adapter.require(template['model'] == fid and source['model'] == fid, fid + ': incorrect model source')
        direction = orientation_model(orientation, fid)
        transform = source_to_observed(template, source, fid)
        source_docs[fid], templates[fid] = source, template
        source_frame = f'saved-surface/{template["source_task"]}/{fid}'
        observed_frame = f'observed-cover-holes/{template["source_task"]}/{fid}'
        S = np.asarray(station_source['station_T_boxes'][fid], float).copy()
        S[:3, :3] = STATION_R_OBSERVED
        models[fid] = {'template_path': template_path, 'source_path': source_path,
                       'source_frame': source_frame, 'observed_frame': observed_frame,
                       'observed_T_source': transform, 'station_T_observed': S,
                       'direction_evidence': direction}
        math_evidence['models'][fid] = {
            'source_surface': evidence(source_path), 'observed_template': evidence(template_path),
            'source_frame_id': source_frame, 'observed_frame_id': observed_frame,
            'same_source_capture': str(Path(source['source_scan']).resolve()),
            'camera_T_observed': template['camera_T_box'], 'camera_T_source': source['camera_T_box'],
            'observed_T_source': transform.tolist(),
            'reconstructed_camera_T_source_max_abs_residual': float(np.max(np.abs(
                np.asarray(template['camera_T_box']) @ transform - np.asarray(source['camera_T_box'])))),
            'registration_scope': 'Mathematical change of frame for the same RGB-D capture; no motor-to-CAD registration',
        }
        tracked.extend([template_path, source_path])
    sources_before = {str(p.resolve()): sha(p) for p in tracked}
    inputs.mkdir(parents=True, exist_ok=True)
    math_path = inputs / names[0]
    write_json(math_path, math_evidence)
    manifest = {'schema_version': adapter.SCHEMA_VERSION, 'status': 'offline_design_candidate',
                'physical_motor_registration_confirmed': False, 'models': {}}
    station = copy.deepcopy(station_source)
    station['intrinsic_geometry_confirmed'] = False
    station['physical_motor_registration_confirmed'] = False
    station['confirmation_scope'] = ('Offline design candidate: independent observed terminal/surface geometry and '
                                     'photo-based orientation; archived CAD foot-to-box translations remain nominal and unaccepted.')
    station['geometry_scope'] = station['confirmation_scope']
    station['source_station_geometry'] = evidence(args.station_geometry)
    station['source_observed_orientation_evidence'] = evidence(args.orientation_evidence)
    station['station_T_boxes_frame_semantics'] = 'Observed cover-hole semantic frames, not the old CAD box frames'
    station['observed_orientation_diagnostics'] = {}
    station['limitations'] = list(station.get('limitations', [])) + TRANSLATION_LIMITS + [
        'JL100/JL132 box surface solids now come from their own same-capture RGB-D height fields; the conflicting old CAD shells are not reused.',
        'JM160 same-capture height field and all seven source cylinders are rigidly registered together; all missing-depth occupancy is retained.',
        'The shared motor body, auxiliary envelopes, shaft and physical foot datums remain exactly as in the archived CAD station model.',
    ]
    for fid, info in models.items():
        source_info = evidence(info['source_path'], frame_id=info['source_frame'], kind='scan')
        source_registration = {
            'status': 'verified', 'source_frame_id': info['source_frame'],
            'observed_frame_id': info['observed_frame'], 'observed_T_source': info['observed_T_source'].tolist(),
            'method': 'Exact same-capture frame change: inverse(camera_T_observed) @ camera_T_source',
            'registration_scope': 'Coordinate identity within the named capture, not physical motor-to-CAD fitting',
            'evidence': [evidence(math_path), evidence(info['template_path']), evidence(info['source_path'])],
        }
        station_registration = {
            'status': 'nominal_cad_translation_observed_orientation', 'physical_registration_confirmed': False,
            'station_frame_id': 'shared-four-foot-slot-centre-level-table', 'observed_frame_id': info['observed_frame'],
            'station_T_observed': info['station_T_observed'].tolist(),
            'method': ('Proposed observed +X cable side maps to CAD station +Y; observed +Z points into the upward box; '
                       'box-centre translation is unchanged nominal whole-motor CAD'),
            'orientation_evidence': [evidence(args.orientation_evidence), evidence(info['template_path'])],
            'translation_evidence': [evidence(args.station_geometry)],
            'translation_limitations': TRANSLATION_LIMITS,
            'orientation_evidence_details': info['direction_evidence'],
        }
        spec = adapter.model_from_template(fid, info['template_path'], source_info, source_registration,
                station_registration, MATING_ORDER[fid], OTHER_ORDER[fid], [evidence(args.mating_evidence)])
        manifest['models'][fid] = spec
        station['station_T_boxes'][fid] = info['station_T_observed'].tolist()
        # Preserve the existing CAD datums verbatim; their existing physical
        # registration flag is already false. New frame evidence is separate.
        adapter.require(station['model_datums'][fid].get('physical_motor_to_box_registration_confirmed') is False,
                        fid + ': source CAD datum must not claim physical registration')
        features = {x['id']: np.asarray(x['xyz_box_mm'], float) for x in templates[fid]['features']}
        observed_direction = features[MATING_ORDER[fid][-1]] - features[MATING_ORDER[fid][0]]
        station_direction = STATION_R_OBSERVED @ observed_direction
        shaft = np.asarray(station['model_datums'][fid]['shaft_direction_station'], float)
        angle = float(np.degrees(np.arccos(np.clip(abs(station_direction @ shaft) /
                      (np.linalg.norm(station_direction) * np.linalg.norm(shaft)), 0, 1))))
        xy = station_direction.copy(); xy[2] = 0
        xy_angle = float(np.degrees(np.arccos(np.clip(abs(xy @ shaft) / np.linalg.norm(xy), 0, 1))))
        station['observed_orientation_diagnostics'][fid] = {
            'mating_feature_number_order': MATING_ORDER[fid], 'other_feature_number_order': OTHER_ORDER[fid],
            'observed_mating_row_station_direction': (station_direction / np.linalg.norm(station_direction)).tolist(),
            'row_vs_shaft_parallel_angle_deg_3d': angle, 'row_vs_shaft_parallel_angle_deg_xy': xy_angle,
            'orientation_candidate_only': True, 'physical_foot_registration_confirmed': False,
            'interpretation': ('User confirms JL100 three-stud row is parallel to shaft; small observed deviation is retained, not forced to zero.'
                               if fid == 'JL100' else 'Observed local row is mapped using photo cable-side semantics; whole-motor foot registration remains nominal.')}
    # Body, external fittings and physical mounting datums must not move while
    # fixing the independent box frame.
    for key in ('motor_envelopes', 'motor_aux_envelopes', 'model_datums', 'station_frame'):
        adapter.require(station[key] == station_source[key], key + ': CAD motor/foot geometry changed')
    station['constraint_validation'] = validate_station_geometry(station)
    manifest_path, station_path = inputs / names[1], inputs / names[2]
    write_json(manifest_path, manifest)
    write_json(station_path, station)
    build_result = adapter.build(manifest_path, output)
    adapter.require(not build_result['errors'], 'Registered geometry rejected: ' + str(build_result['errors']))
    final_sources = {str(p.resolve()): sha(p) for p in tracked}
    adapter.require(final_sources == sources_before, 'An input changed while preparing the candidate')
    receipt = {
        'status': 'offline_design_candidate_prepared', 'physical_motor_registration_confirmed': False,
        'manifest': evidence(manifest_path), 'station_geometry': evidence(station_path),
        'registration_math': evidence(math_path), 'registered_frames': evidence(output / 'registered_frames.json'),
        'source_sha256_at_start': sources_before, 'source_sha256_at_finish': final_sources,
        'orientation_diagnostics': station['observed_orientation_diagnostics'],
        'preserved_geometry_keys': ['motor_envelopes', 'motor_aux_envelopes', 'model_datums', 'station_frame'],
        'collision_preservation': {fid: json.loads((output / (fid + '_collision.json')).read_text())['preservation'] for fid in adapter.MODELS},
        'hardware_actions': False, 'planning_performed': False, 'limitations': TRANSLATION_LIMITS,
    }
    write_json(inputs / names[3], receipt)
    print(json.dumps({'status': receipt['status'], 'manifest': str(manifest_path),
                      'station_geometry': str(station_path), 'registered_geometry': str(output),
                      'orientation_diagnostics': receipt['orientation_diagnostics']}, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station-geometry', type=Path, default=OLD_STATION)
    parser.add_argument('--templates', type=Path, default=BASE / 'direction-audit/templates')
    parser.add_argument('--scans', type=Path, default=BASE / 'observed-scan-geometry')
    parser.add_argument('--scan160', type=Path, default=OLD_SCAN160)
    parser.add_argument('--orientation-evidence', type=Path, required=True)
    parser.add_argument('--mating-evidence', type=Path, default=BASE / 'archive/mating-audit/recorded_fixture_mating.json')
    parser.add_argument('--inputs', type=Path, default=BASE / 'planning-inputs')
    parser.add_argument('--output', type=Path, default=BASE / 'registered-geometry')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
