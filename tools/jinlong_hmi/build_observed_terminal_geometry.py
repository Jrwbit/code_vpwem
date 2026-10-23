#!/usr/bin/env python3
"""Register independently observed terminals and archived box solids offline.

This adapter deliberately has no fixture/tool input.  A terminal template and
a box mesh do not establish their relative frame: both ``observed_T_source``
and ``station_T_observed`` require explicit, evidence-backed registrations.
Missing registrations produce diagnostics, never a solver-ready collision file.

Input schema (all translations, points, radii and lengths are in metres)::

  {"schema_version": "observed-terminal-geometry/v1", "models": {
    "JL100": {
      "observed_frame": {"id": "scan/task/semantic-box", "definition": "..."},
      "source_geometry": {"path": "JL100_collision.json", "sha256": "...",
                          "frame_id": "cad/cover-hole-frame", "kind": "cad"},
      "geometry_registration": {
        "status": "verified", "source_frame_id": "cad/cover-hole-frame",
        "observed_frame_id": "scan/task/semantic-box",
        "observed_T_source": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "method": "Independent matched physical features, not tool fitting",
        "evidence": [{"path": "registration.json", "sha256": "..."}]},
      "station_registration": {
        "status": "verified", "station_frame_id": "common-foot-datum",
        "observed_frame_id": "scan/task/semantic-box", "station_T_observed": [],
        "method": "Measured box frame relative to physical foot datum",
        "evidence": [{"path": "station-registration.json", "sha256": "..."}]},
      "observation": {
        "id": "same-capture-id", "frame_id": "scan/task/semantic-box",
        "basis": "rgbd", "independent_of_tool": true,
        "evidence": [{"path": "terminal-observations.json", "sha256": "..."}]},
      "terminals": [{
        "id": "observed-feature-id", "kind": "stud", "role": "mating",
        "collision_name": "term_JL100_stud_1",
        "position_m": [0,0,0], "axis": [0,0,1], "radius_m": 0.003,
        "length_m": 0.030, "depth_status": "measured",
        "dimension_source": "M6 radius/length remain model assumptions",
        "source_terminal_name": "term_JM160_stud_1"}],
      "limitations": []}}}

Exactly six studs (three mating, three other) and one ground are required. Every
feature has an explicit ``collision_name``; electrical/photo array ordering is
not assumed to match socket numbering (JL132 in particular reverses both rows).
``basis`` may be rgbd, survey, manufacturer_drawing, or image_ray_plane;
``depth_status`` may be measured or assumed.  Assumed depths remain prominently
labelled in outputs and cannot establish physical insertion.  For a scan source,
every original named terminal cylinder must be preserved exactly after the same
rigid registration: the height field has terminal-sized holes and cannot safely
be paired with terminals from another capture. ``source_terminal_name`` provides
that identity; it is optional for CAD sources without terminal cylinders.

All evidence paths are relative to the input JSON, unless absolute. SHA256s are
checked before use. Output parts retain source indices and unknown-voxel flags.
The adapter never changes the source meshes, tool, camera, table, motor body,
clearance limits or unknown scan occupancy. No hardware or network access.

Only ``station_registration`` also permits the explicitly qualified status
``nominal_cad_translation_observed_orientation`` for offline design searches.
It requires ``physical_registration_confirmed: false``, separately hashed
``orientation_evidence`` and ``translation_evidence`` lists, and a nonempty
``translation_limitations`` list. The orientation has observation evidence;
the translation is a nominal CAD design assumption, not measured motor-to-foot
registration. Such outputs are labelled ``offline_design_candidate``. This
exception never applies to ``geometry_registration``: the source collision
solids must still have an explicit verified mapping into the observation frame.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023/registered-geometry'
SCHEMA_VERSION = 'observed-terminal-geometry/v1'
MODELS = ('JM160', 'JL132', 'JL100')
BOX_FACES = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
             [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
             [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]


class GeometryInputError(ValueError):
    """The supplied evidence does not establish the requested registration."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise GeometryInputError(message)


def rigid_matrix(value, label):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GeometryInputError(f'{label}: expected a finite 4x4 rigid transform') from exc
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(),
            f'{label}: expected a finite 4x4 rigid transform, translation in metres')
    require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0),
            f'{label}: invalid homogeneous bottom row')
    rotation = matrix[:3, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) and
            abs(np.linalg.det(rotation) - 1) < 1e-7,
            f'{label}: rotation must be proper and orthonormal (no reflection)')
    return matrix


def vector(value, label):
    try:
        result = np.asarray(value, float)
    except (TypeError, ValueError) as exc:
        raise GeometryInputError(f'{label}: expected three finite coordinates') from exc
    require(result.shape == (3,) and np.isfinite(result).all(),
            f'{label}: expected three finite coordinates')
    return result


def evidence_files(entries, base, label):
    require(isinstance(entries, list) and bool(entries), f'{label}: missing independent evidence')
    checked = []
    for entry in entries:
        require(isinstance(entry, dict) and bool(entry.get('path')),
                f'{label}: evidence must identify a file')
        path = (base / entry['path']).resolve()
        require(path.is_file(), f'{label}: missing evidence file {path}')
        digest = sha256(path)
        require(entry.get('sha256') == digest, f'{label}: evidence SHA256 mismatch: {path}')
        checked.append(dict(entry, path=str(path), sha256=digest))
    return checked


def registration(spec, key, matrix_key, base, expected_frames):
    value = spec.get(key)
    require(isinstance(value, dict), f'{key}: missing registration; frame equality cannot be assumed')
    nominal = key == 'station_registration' and value.get('status') == 'nominal_cad_translation_observed_orientation'
    require(value.get('status') == 'verified' or nominal, f'{key}: registration is not verified')
    for field, expected in expected_frames.items():
        require(value.get(field) == expected, f'{key}: {field} does not match {expected!r}')
    require(bool(value.get('method')), f'{key}: missing independent registration method')
    matrix = rigid_matrix(value.get(matrix_key), f'{key}.{matrix_key}')
    if nominal:
        require(value.get('physical_registration_confirmed') is False,
                f'{key}: nominal CAD translation must explicitly set physical_registration_confirmed=false')
        limitations = value.get('translation_limitations')
        require(isinstance(limitations, list) and limitations and
                all(isinstance(x, str) and x.strip() for x in limitations),
                f'{key}: nominal CAD translation requires explicit translation_limitations')
        orientation = evidence_files(value.get('orientation_evidence'), base, key + '.orientation_evidence')
        translation = evidence_files(value.get('translation_evidence'), base, key + '.translation_evidence')
        return matrix, dict(value, orientation_evidence=orientation, translation_evidence=translation,
                            evidence=orientation + translation)
    checked = evidence_files(value.get('evidence'), base, key)
    return matrix, dict(value, evidence=checked)


def terminal_geometry(fid, spec):
    observation = spec.get('observation', {})
    require(observation.get('independent_of_tool') is True,
            'observation: terminal coordinates must be independent of the tool')
    require(observation.get('basis') in {'rgbd', 'survey', 'manufacturer_drawing', 'image_ray_plane'},
            'observation: unsupported coordinate source (fixture-derived points are prohibited)')
    require(bool(observation.get('id')), 'observation: capture/measurement identity is required')
    require(observation.get('frame_id') == spec['observed_frame']['id'],
            'observation: points and observed frame do not share a frame ID')
    features = spec.get('terminals', [])
    require(isinstance(features, list) and len(features) == 7,
            'terminals: require six independently observed studs and one ground')
    roles = [(x.get('kind'), x.get('role')) for x in features]
    require(roles.count(('stud', 'mating')) == 3 and roles.count(('stud', 'other')) == 3 and
            roles.count(('ground', 'ground')) == 1, 'terminals: expected 3 mating + 3 other studs + 1 ground')
    require(all(isinstance(x.get('id'), str) and x['id'] for x in features) and
            len({x['id'] for x in features}) == 7, 'terminals: each feature needs a unique source identity')
    allowed_names = {'mating': {f'term_{fid}_stud_{i}' for i in range(1, 4)},
                     'other': {f'term_{fid}_other_{i}' for i in range(1, 4)},
                     'ground': {f'term_{fid}_gnd'}}
    require(len({x.get('collision_name') for x in features}) == 7,
            'terminals: collision identities must be explicit and unique; do not assume photo array order')
    terminals = []
    for feature in features:
        label = 'terminal ' + feature['id']
        origin = vector(feature.get('position_m'), label + '.position_m')
        axis = vector(feature.get('axis'), label + '.axis')
        require(abs(np.linalg.norm(axis) - 1) < 1e-7, label + ': axis must be unit length')
        radius, length = feature.get('radius_m'), feature.get('length_m')
        require(isinstance(radius, (float, int)) and np.isfinite(radius) and radius > 0 and
                isinstance(length, (float, int)) and np.isfinite(length) and length > 0,
                label + ': radius and length must be finite and positive, in metres')
        require(feature.get('depth_status') in {'measured', 'assumed'}, label + ': identify measured/assumed depth')
        require(bool(feature.get('dimension_source')), label + ': cylinder dimension provenance is required')
        role = feature['role']
        require(feature.get('collision_name') in allowed_names[role], label + ': collision name does not match role/model')
        terminals.append({'name': feature['collision_name'], 'a': origin.tolist(),
                          'b': (origin + axis * length).tolist(), 'radius': float(radius),
                          'observed_feature': copy.deepcopy(feature),
                          'observation_id': observation['id'], 'center_source': feature['id'],
                          'dimension_source': feature['dimension_source']})
    return terminals


def transform_points(points, transform):
    return np.asarray(points, float) @ transform[:3, :3].T + transform[:3, 3]


def preserve_scan_terminals(source, terminals, transform):
    """Do not leave holes by combining a height field with a different template."""
    originals = source.get('measured_terminals', [])
    require(len(originals) == 7, 'scan: source must contain all six observed studs and ground')
    names = {x['name'] for x in originals}
    replacements = {x['observed_feature'].get('source_terminal_name'): x for x in terminals}
    require(set(replacements) == names and len(replacements) == 7,
            'scan: every source terminal needs an explicit, one-to-one observed feature identity')
    for old in originals:
        new = replacements[old['name']]
        expected = transform_points([old['a'], old['b']], transform)
        require(np.allclose(expected, [new['a'], new['b']], rtol=0, atol=1e-9) and
                abs(float(old['radius']) - new['radius']) < 1e-12,
                'scan: registered terminal differs from its source cylinder; '
                f'cannot reuse terminal-excluded voxels for {old["name"]}')


def prepare_model(fid, spec, base):
    require(fid in MODELS, f'unsupported motor model {fid}')
    frame = spec.get('observed_frame', {})
    require(bool(frame.get('id')) and bool(frame.get('definition')),
            'observed_frame: explicit semantic frame ID and definition required')
    source_info = spec.get('source_geometry', {})
    require(source_info.get('kind') in {'cad', 'scan'}, 'source_geometry: kind must be cad or scan')
    require(bool(source_info.get('frame_id')), 'source_geometry: source frame ID is required')
    require(source_info['frame_id'] != frame['id'],
            'source and observed frame IDs must remain distinct, even for an explicitly registered identity transform')
    # Diagnose missing registrations before trying to consume geometry.
    T, geometry_registration = registration(spec, 'geometry_registration', 'observed_T_source', base,
            {'source_frame_id': source_info['frame_id'], 'observed_frame_id': frame['id']})
    station = spec.get('station_registration', {})
    require(bool(station.get('station_frame_id')), 'station_registration: missing common foot-datum frame ID')
    S, station_registration = registration(spec, 'station_registration', 'station_T_observed', base,
            {'observed_frame_id': frame['id']})
    source_path = evidence_files([source_info], base, 'source_geometry')[0]
    source = json.loads(Path(source_path['path']).read_text())
    require(source.get('model') == fid and source.get('units') == 'metres',
            'source_geometry: wrong model or units')
    observation = dict(spec.get('observation', {}))
    terminals = terminal_geometry(fid, spec)
    observation['evidence'] = evidence_files(observation.get('evidence'), base, 'observation')
    if source_info['kind'] == 'scan':
        preserve_scan_terminals(source, terminals, T)
    else:
        require(not source.get('measured_terminals'),
                'CAD source must contain box solid only, not fixture-derived terminal cylinders')
    parts = []
    for index, part in enumerate(source.get('parts', [])):
        vertices = np.asarray(part['vertices'], float)
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(),
                f'source_geometry.parts[{index}]: invalid vertices')
        parts.append(dict(copy.deepcopy(part), vertices=transform_points(vertices, T).tolist(),
                          source_part_index=index))
    for index, box in enumerate(source.get('boxes', [])):
        center = vector(box.get('center'), f'source box {index}.center')
        half = vector(box.get('half_size'), f'source box {index}.half_size')
        require(np.all(half > 0), f'source box {index}: invalid half size')
        vertices = center + np.asarray(list(itertools.product((-1, 1), repeat=3))) * half
        # Exact rigid box, not its enlarged axis-aligned bounding box.
        parts.append({'vertices': transform_points(vertices, T).tolist(), 'faces': BOX_FACES,
                      'unknown': bool(box.get('unknown', False)), 'source_box_index': index,
                      'source_box': copy.deepcopy(box)})
    require(bool(parts), 'source_geometry: empty box solids cannot establish collision clearance')
    unknown_before = sum(bool(x.get('unknown', False)) for x in source.get('boxes', []))
    unknown_after = sum(bool(x.get('unknown', False)) for x in parts)
    require(unknown_after >= unknown_before, 'unknown scan occupancy was lost')
    assumed_depth = any(x['observed_feature']['depth_status'] == 'assumed' for x in terminals)
    nominal_station = station_registration['status'] == 'nominal_cad_translation_observed_orientation'
    physical_confirmed = station_registration.get('physical_registration_confirmed') is True and not nominal_station
    return {
        'model': fid, 'units': 'metres', 'schema_version': SCHEMA_VERSION,
        'status': ('offline_design_candidate' if nominal_station else
                   'offline_registered_orientation_candidate_assumed_depth' if assumed_depth
                   else 'offline_registered_observed_geometry_not_site_acceptance'),
        'physical_motor_registration_confirmed': physical_confirmed,
        'geometry_scope': ('Observed terminal direction and source-geometry frame mapping; '
                           'motor-to-foot translation is nominal CAD, physical registration precision unaccepted'
                           if nominal_station else 'Explicit source and station frame mappings with recorded evidence; no site acceptance'),
        'frame': frame['definition'], 'observed_frame': copy.deepcopy(frame),
        'source_geometry': source_path, 'geometry_registration': geometry_registration,
        'station_registration': station_registration, 'station_T_observed': S.tolist(),
        'observation': observation, 'parts': parts, 'boxes': [], 'measured_terminals': terminals,
        'preservation': {'source_convex_parts': len(source.get('parts', [])),
                         'source_scan_boxes': len(source.get('boxes', [])),
                         'source_unknown_boxes': unknown_before, 'output_unknown_parts': unknown_after,
                         'source_unknown_columns': source.get('unknown_columns'),
                         'solid_transform_only': True, 'scan_terminal_cylinders_preserved': source_info['kind'] == 'scan'},
        'source_geometry_metadata': {k: v for k, v in source.items() if k not in {'parts', 'boxes', 'measured_terminals'}},
        'limitations': list(source.get('limitations', [])) + list(spec.get('limitations', [])) + [
            'No fixture-derived workpiece points are used; dimensions still carry their stated source assumptions.',
            'Registration provenance is checked by file hash; this adapter does not certify physical registration accuracy.',
            'Registered geometry alone does not establish reachability, complete route, physical insertion or electrical acceptance.',
        ] + (['At least one terminal depth is assumed; this model cannot certify physical mating.'] if assumed_depth else []) +
            (['Motor-to-foot translation is a nominal CAD design assumption, not a measured physical registration.'] +
             station_registration['translation_limitations'] if nominal_station else []),
    }


def model_from_template(fid, template_path, source_geometry, geometry_registration,
                        station_registration, mating_feature_ids, other_feature_ids,
                        mating_selection_evidence):
    """Package the independent RGB-D template without re-estimating any frame.

    Registrations and the three active feature identities are mandatory caller
    inputs. The ordered ``mating_feature_ids`` and ``other_feature_ids`` map to
    ``stud_1..3`` and ``other_1..3`` respectively; photo/electrical order is not
    automatically socket order. ``mating_selection_evidence`` identifies the real feedback/image
    comparison that selects a row; current tool geometry must not generate the
    motor points. Scan cylinder positions retain the original conservative P5
    top and identical excluded footprint; refreshed pixel observations are kept
    alongside them for audit rather than silently moving those cylinders.
    """
    template_path = Path(template_path).resolve()
    template = json.loads(template_path.read_text())
    require(template.get('model') == fid and template.get('tool_geometry_used_for_measurement') is False,
            'template must be the named independent motor observation')
    features = template.get('features', [])
    require(len(features) == 7 and len({x['id'] for x in features}) == 7,
            'template must contain six identified studs and one ground')
    require(len(mating_feature_ids) == 3 and len(set(mating_feature_ids)) == 3 and
            set(mating_feature_ids) <= {x['id'] for x in features if x['id'] != 'ground'},
            'explicit identities of three actually mating observed studs are required')
    require(len(other_feature_ids) == 3 and len(set(other_feature_ids)) == 3 and
            set(other_feature_ids).isdisjoint(mating_feature_ids) and
            set(other_feature_ids) | set(mating_feature_ids) == {x['id'] for x in features if x['id'] != 'ground'},
            'explicit numbered identities of all other studs are required')
    require(bool(mating_selection_evidence), 'mating row selection requires independent evidence')
    source_path = Path(source_geometry['path']).resolve()
    require(sha256(source_path) == source_geometry['sha256'], 'source geometry SHA256 mismatch')
    source = json.loads(source_path.read_text())
    T = rigid_matrix(geometry_registration.get('observed_T_source'), 'observed_T_source')
    source_terminals = source.get('measured_terminals', [])
    is_scan = source_geometry['kind'] == 'scan'
    if is_scan:
        require(template.get('source_scan') and source.get('source_scan') and
                Path(template['source_scan']).resolve() == Path(source['source_scan']).resolve(),
                'scan terminal template and height field must come from the same capture')
        require(np.allclose(T, np.linalg.inv(rigid_matrix(template['camera_T_box'], 'template.camera_T_box')) @
                            rigid_matrix(source.get('camera_T_box'), 'scan.camera_T_box'), atol=1e-9, rtol=0),
                'observed_T_source does not match the same-capture camera-frame registration')
    packaged = []
    for feature in features:
        ground = feature['id'] == 'ground'
        item = {'id': feature['id'], 'kind': 'ground' if ground else 'stud',
                'role': 'ground' if ground else 'mating' if feature['id'] in mating_feature_ids else 'other',
                'depth_status': 'measured', 'independent_observed_feature': copy.deepcopy(feature)}
        if ground:
            item['collision_name'] = f'term_{fid}_gnd'
        elif feature['id'] in mating_feature_ids:
            item['collision_name'] = f'term_{fid}_stud_{mating_feature_ids.index(feature["id"]) + 1}'
        else:
            item['collision_name'] = f'term_{fid}_other_{other_feature_ids.index(feature["id"]) + 1}'
        if is_scan:
            matches = [old for old in source_terminals if old.get('center_source') == feature.get('legacy_id')]
            require(len(matches) == 1, f'no unique original scan cylinder for feature {feature["id"]}')
            original = matches[0]
            a, b = transform_points([original['a'], original['b']], T)
            length = float(np.linalg.norm(b - a))
            item.update(position_m=a.tolist(), axis=((b - a) / length).tolist(),
                        radius_m=original['radius'], length_m=length,
                        source_terminal_name=original['name'],
                        dimension_source=original.get('radius_source', 'Inherited cylinder dimensions, not measured'),
                        collision_top_source='Original same-capture outward P5 cylinder transformed with all voxels',
                        refreshed_template_center_delta_mm=(a * 1000 - np.asarray(feature['xyz_box_mm'])).tolist())
        else:
            position = np.asarray(feature['xyz_box_mm'], float) / 1000
            cap = feature.get('cap_surface_depth_mm_percentiles_5_50_95',
                              feature.get('depth_below_rim_mm_percentiles_5_50_95'))
            require(isinstance(cap, list) and len(cap) == 3,
                    f'missing measured outward cap depth for {feature["id"]}')
            position[2] = cap[0] / 1000
            item.update(position_m=position.tolist(), axis=[0., 0., 1.],
                        radius_m=.004 if ground else .003, length_m=.008 if ground else .030,
                        dimension_source='Inherited M6 stud / D8 ground-cap cylinder and 30/8mm length assumptions; not ROI measurements',
                        collision_top_source='Independent cap ROI outward P5 depth (centre XY retained)')
        packaged.append(item)
    frame_id = geometry_registration['observed_frame_id']
    return {
        'observed_frame': {'id': frame_id, 'definition': template['frame'],
                           'camera_T_observed': template['camera_T_box']},
        'source_geometry': copy.deepcopy(source_geometry),
        'geometry_registration': copy.deepcopy(geometry_registration),
        'station_registration': copy.deepcopy(station_registration),
        'observation': {'id': template['source_task'], 'frame_id': frame_id,
                        'basis': 'rgbd', 'independent_of_tool': True,
                        'evidence': [{'path': str(template_path), 'sha256': sha256(template_path)}] +
                                    copy.deepcopy(mating_selection_evidence),
                        'mating_feature_ids': list(mating_feature_ids),
                        'other_feature_ids': list(other_feature_ids),
                        'mating_selection': 'Caller-supplied actual path/feedback to independent image comparison',
                        'template_source_sha256': template.get('source_sha256', {})},
        'terminals': packaged,
        'limitations': list(template.get('limitations', [])),
    }


def build(input_path, output):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    document = json.loads(input_path.read_text())
    require(document.get('schema_version') == SCHEMA_VERSION, 'wrong or missing schema_version')
    models = document.get('models', {})
    require(isinstance(models, dict) and bool(models), 'models must be a nonempty dictionary')
    # Reject reuse rather than leaving old passing collision files beside new
    # failure diagnostics. The caller can choose a fresh run directory.
    require(not any((output / (fid + '_collision.json')).exists() for fid in MODELS),
            'output already contains collision files; use a fresh directory to preserve historical evidence')
    prepared, errors = {}, {}
    for fid, spec in models.items():
        try:
            prepared[fid] = prepare_model(fid, spec, input_path.parent)
        except GeometryInputError as exc:
            errors[fid] = str(exc)
    station_ids = {x['station_registration']['station_frame_id'] for x in prepared.values()}
    if len(station_ids) > 1:
        errors['common_station'] = 'models do not share one physical foot-datum frame'
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {'status': 'rejected_incomplete_registration' if errors else 'registered_geometry_written',
                   'input': str(input_path), 'input_sha256': sha256(input_path),
                   'adapter_sha256': sha256(__file__), 'errors': errors,
                   'written_models': [] if errors else list(prepared), 'hardware_actions': False}
    # Atomic validation of the entire bundle: never write a partial model set.
    if not errors:
        for fid, result in prepared.items():
            result['input_manifest_sha256'] = diagnostics['input_sha256']
            result['adapter_sha256'] = diagnostics['adapter_sha256']
            (output / (fid + '_collision.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        frames = {'schema_version': SCHEMA_VERSION, 'station_frame_id': next(iter(station_ids)),
                  'station_T_observed': {fid: result['station_T_observed'] for fid, result in prepared.items()},
                  'collision_files_sha256': {fid: sha256(output / (fid + '_collision.json')) for fid in prepared},
                  'input_manifest_sha256': diagnostics['input_sha256'],
                  'physical_motor_registration_confirmed': all(x['physical_motor_registration_confirmed'] for x in prepared.values()),
                  'geometry_scope_by_model': {fid: x['geometry_scope'] for fid, x in prepared.items()}}
        (output / 'registered_frames.json').write_text(json.dumps(frames, ensure_ascii=False, indent=2) + '\n')
    (output / 'registration_diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
    return diagnostics


def bind_scene(scene, fid, world_T_station, geometry_dir=DEFAULT_OUTPUT):
    """Bind a registered observed frame; preserve every non-box scene obstacle.

    Caller supplies only the shared world_T_station. The semantic box pose is
    computed from the verified station_T_observed, never a CAD-frame matrix.
    Return a copy; no production scene is mutated.
    """
    from build_measured_box_geometry import modify_scene as original_modify_scene
    geometry_dir = Path(geometry_dir)
    frames = json.loads((geometry_dir / 'registered_frames.json').read_text())
    path = geometry_dir / (fid + '_collision.json')
    require(sha256(path) == frames['collision_files_sha256'].get(fid), 'registered collision file SHA256 mismatch')
    doc = json.loads(path.read_text())
    require(doc.get('schema_version') == SCHEMA_VERSION and doc.get('model') == fid,
            'scene binding requires the registered observed geometry format')
    W = rigid_matrix(world_T_station, 'world_T_station')
    S = rigid_matrix(frames['station_T_observed'][fid], 'station_T_observed')
    require(np.allclose(S, doc['station_T_observed'], rtol=0, atol=1e-12), 'registered frame manifest mismatch')
    world_T_observed = W @ S
    if 'box_T_base_link' in scene:
        existing = rigid_matrix(scene['box_T_base_link'], 'scene.box_T_base_link')
        require(np.allclose(existing, world_T_observed, rtol=0, atol=1e-9),
                'scene box_T_base_link differs from registered world_T_observed; '
                'rebuild targets in the observed frame before binding collision geometry')
    result = copy.deepcopy(scene)
    result['box_T_base_link'] = world_T_observed.tolist()
    result['world_T_station'] = W.tolist()
    result['observed_box_frame_id'] = doc['observed_frame']['id']
    result['common_station_frame_id'] = frames['station_frame_id']
    result['physical_motor_registration_confirmed'] = doc['physical_motor_registration_confirmed']
    result['observed_geometry_scope'] = doc['geometry_scope']
    result = original_modify_scene(result, fid, geometry_dir)
    require(result.get('terminals') is False, 'fixture-derived terminals must remain disabled')
    return result


def modify_scene(scene, fid, geometry_dir=DEFAULT_OUTPUT):
    """Simulator adapter with the original function signature.

    ``world_T_station`` is mandatory. A bare ``box_T_base_link`` provides no
    evidence whether it represents the CAD or the observed semantic frame.
    """
    require('world_T_station' in scene,
            'scene requires explicit world_T_station; a bare CAD/box pose cannot establish the observed frame')
    registered = bind_scene(scene, fid, scene['world_T_station'], geometry_dir)
    # Match the existing adapter's in-place API; both old Study callers ignore
    # the return value. bind_scene remains available as a pure copy operation.
    scene.clear()
    scene.update(registered)
    return scene


def build_model(scene, fid, original_build):
    """Compile with the existing measured geometry hook and retained tool solids."""
    from build_measured_box_geometry import build_model as original_build_model
    path = Path(scene['model_specific_box_collision'])
    doc = json.loads(path.read_text())
    require(doc.get('schema_version') == SCHEMA_VERSION and doc.get('model') == fid,
            'build_model requires registered observed geometry')
    require(scene.get('terminals') is False and
            scene.get('observed_box_frame_id') == doc['observed_frame']['id'],
            'build_model requires observed frame binding with fixture-derived terminals disabled')
    return original_build_model(scene, fid, original_build)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.input, args.output)
    except GeometryInputError as exc:
        parser.exit(2, str(exc) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
