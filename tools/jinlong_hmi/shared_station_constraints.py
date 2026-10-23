"""Independent geometric invariants for one fixed, level motor station.

These checks constrain the search space. They do not establish IK, collision,
camera coverage, measured CAD accuracy, or physical insertion acceptance.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


FIXTURES = ("JM160", "JL132", "JL100")
TRANSFORM_TOL = 1e-8
FOOT_PLANE_TOL_M = 1e-4


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _transform(value, label):
    T = np.asarray(value, dtype=float)
    _require(T.shape == (4, 4) and np.isfinite(T).all(), label + ': finite 4x4 transform required')
    _require(np.allclose(T[3], [0, 0, 0, 1], atol=TRANSFORM_TOL, rtol=0), label + ': invalid homogeneous row')
    R = T[:3, :3]
    _require(np.allclose(R.T @ R, np.eye(3), atol=TRANSFORM_TOL, rtol=0)
             and abs(np.linalg.det(R) - 1) <= TRANSFORM_TOL, label + ': proper rigid rotation required')
    return T


def _same(actual, expected, label):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    _require(actual.shape == expected.shape and np.isfinite(actual).all()
             and np.allclose(actual, expected, atol=TRANSFORM_TOL, rtol=0), label)


def _one_obstacle(doc, name):
    matches = [o for o in doc['obstacles'] if o['name'] == name]
    _require(len(matches) == 1, 'Exactly one ' + name + ' obstacle is required')
    return matches[0]


def validate_station_geometry(geometry):
    """Reject intrinsic geometry that cannot sit on the one level foot plane."""
    _require(set(geometry['station_T_boxes']) == set(FIXTURES), 'All three fixed intrinsic box transforms are required')
    _require(set(geometry['motor_envelopes']) == set(FIXTURES), 'All three intrinsic motor envelopes are required')
    rows = {}
    datums = geometry.get('model_datums')
    if geometry.get('intrinsic_geometry_confirmed', False):
        _require(datums is not None and set(datums) == set(FIXTURES),
                 'Confirmed CAD geometry requires all three independent foot/shaft datums')
    if 'motor_aux_envelopes' in geometry:
        _require(set(geometry['motor_aux_envelopes']) == set(FIXTURES), 'Auxiliary motor geometry requires all three model lists')
    shaft_reference = None
    for fid in FIXTURES:
        box = _transform(geometry['station_T_boxes'][fid], fid + ' intrinsic box')
        _same(box[:3, 2], [0, 0, -1], fid + ': box mouth must face upwards')
        _require(box[2, 3] > 0, fid + ': box mouth must be above the common foot plane')
        envelope = geometry['motor_envelopes'][fid]
        motor = _transform(envelope['T'], fid + ' intrinsic motor')
        size = np.asarray(envelope['size'], float)
        _require(size.shape == (3,) and np.isfinite(size).all() and np.all(size > 0), fid + ': invalid motor size')
        corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * size / 2
        bottom = float((corners @ motor[:3, :3].T + motor[:3, 3])[:, 2].min())
        _require(abs(bottom) <= FOOT_PLANE_TOL_M,
                 fid + ': motor envelope must meet the common foot plane without a model-specific riser')
        rows[fid] = {'box_height_above_foot_mm': float(box[2, 3] * 1000),
                     'motor_envelope_bottom_mm': bottom * 1000}
        aux = geometry.get('motor_aux_envelopes', {}).get(fid, [])
        names = [a['name'] for a in aux]
        _require(len(set(names)) == len(names) and all(n.startswith('motor_fitting_') for n in names),
                 fid + ': auxiliary motor geometry must have unique motor_fitting_ names')
        for part in aux:
            _transform(part['T'], part['name'])
            size = np.asarray(part['size'], float)
            _require(size.shape == (3,) and np.isfinite(size).all() and np.all(size > 0), part['name'] + ': invalid size')
        rows[fid]['auxiliary_motor_parts'] = len(aux)
        if datums is not None:
            datum = datums[fid]
            _require(abs(float(datum['foot_datum_z_m'])) <= TRANSFORM_TOL,
                     fid + ': intrinsic foot datum must be the common zero plane')
            shaft = np.asarray(datum['shaft_direction_station'], float)
            _require(shaft.shape == (3,) and np.isfinite(shaft).all()
                     and abs(np.linalg.norm(shaft) - 1) <= TRANSFORM_TOL and abs(shaft[2]) <= TRANSFORM_TOL,
                     fid + ': shaft must be horizontal and have a unit direction')
            if shaft_reference is None:
                shaft_reference = shaft
            _same(shaft, shaft_reference, fid + ': all motor output shafts must share one direction')
            holes = np.asarray(datum['foot_hole_centers_station_m'], float)
            _require(holes.shape == (4, 3) and np.isfinite(holes).all(), fid + ': four foot datum points required')
            _same(holes[:, 2], np.zeros(4), fid + ': all feet must lie on the common zero plane')
            _same(holes[:, :2].mean(0), [0, 0], fid + ': foot-hole rectangle centre must define the shared origin')
    serialized = json.dumps(geometry, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                            default=lambda value: np.asarray(value).tolist())
    return {'passed': True, 'intrinsic_geometry_confirmed': bool(geometry.get('intrinsic_geometry_confirmed', False)),
            'intrinsic_geometry_sha256': hashlib.sha256(serialized.encode()).hexdigest(), 'fixtures': rows}


def validate_station_scene(station, geometry, boxes, doc, *, world_T_camera=None):
    """Reject any per-model shift, yaw, lift, floating motor, or changed table.

    ``boxes`` contains all three world transforms. The active model's scene is
    checked against the same source geometry, independently of target/IK data.
    """
    evidence = validate_station_geometry(geometry)
    station = _transform(station, 'world_T_station')
    _same(station[:3, 2], [0, 0, 1], 'Shared station must be level: only translation and yaw may be searched')
    _require(set(boxes) == set(FIXTURES), 'All three derived world box transforms are required')
    for fid in FIXTURES:
        _same(_transform(boxes[fid], fid + ' world box'), station @ np.asarray(geometry['station_T_boxes'][fid]),
              fid + ': independent box placement is prohibited')
    fid = doc['terminal_fixture']
    _require(fid in FIXTURES, 'Unknown active motor')
    _same(doc['box_T_base_link'], boxes[fid], fid + ': scene box differs from the fixed station')
    motor = _one_obstacle(doc, 'motor_body')
    envelope = geometry['motor_envelopes'][fid]
    expected_motor = station @ np.asarray(envelope['T'])
    _same(motor['T'], expected_motor, fid + ': independent motor placement is prohibited')
    _same(motor['size'], envelope['size'], fid + ': motor envelope must not be resized by the solver')
    auxiliary = geometry.get('motor_aux_envelopes', {}).get(fid, [])
    expected_names = {part['name'] for part in auxiliary}
    actual_names = {o['name'] for o in doc['obstacles'] if o['name'].startswith('motor_fitting_')}
    _require(actual_names == expected_names, fid + ': auxiliary motor fittings must not be omitted or replaced')
    for part in auxiliary:
        actual = _one_obstacle(doc, part['name'])
        _same(actual['T'], station @ np.asarray(part['T']), part['name'] + ': independent fitting placement is prohibited')
        _same(actual['size'], part['size'], part['name'] + ': fitting geometry must not be resized')
    _require(not any(o['name'].startswith('motor_support') for o in doc['obstacles']),
             'Model-specific motor supports are prohibited')
    table = _one_obstacle(doc, 'table')
    _same(table['center_xy'], station[:2, 3], 'The common table must stay at the fixed station XY')
    z = np.asarray(table['z_from_floor'], float)
    _require(z.shape == (2,) and np.isfinite(z).all() and z[0] < z[1], 'Invalid table height interval')
    _require(abs(float(doc['floor_z']) + z[1] - station[2, 3]) <= TRANSFORM_TOL,
             'Tabletop and common foot plane must coincide; model-specific height changes are prohibited')
    meta = doc['shared_motor_station']
    _same(meta['world_T_station'], station, 'Scene metadata changed the common station')
    _same(meta['station_T_box'], geometry['station_T_boxes'][fid], 'Scene metadata changed intrinsic box geometry')
    _require(abs(float(meta['tabletop_world_z_m']) - station[2, 3]) <= TRANSFORM_TOL,
             'Scene metadata changed the common tabletop height')
    _require(meta['no_model_specific_placement_or_height_adjustment'] is True, 'Scene must declare the fixed-station contract')
    camera = _transform(doc['camera_T_base_link'], 'world_T_camera')
    if world_T_camera is not None:
        _same(camera, world_T_camera, 'Model-specific camera repositioning is prohibited')
    # Use the CAD foot locator when available. A protruding shaft can displace
    # the whole-assembly bounding-box centre, so it is not the physical locator.
    datum_xy = station[:2, 3] if geometry.get('model_datums') is not None else expected_motor[:2, 3]
    away = datum_xy
    offset = np.asarray(boxes[fid])[:2, 3] - datum_xy
    radius = float(np.linalg.norm(away))
    _require(radius > TRANSFORM_TOL, 'Cannot define the far side at the robot origin')
    far_side_m = float(np.dot(offset, away / radius))
    _require(far_side_m > TRANSFORM_TOL, fid + ': box must lie on the far side of the motor from the robot origin')
    return {'passed': True, 'fixture': fid, 'geometry': evidence,
            'tabletop_world_z_m': float(station[2, 3]), 'far_side_offset_mm': far_side_m * 1000,
            'far_side_reference': ('robot world XY origin and common CAD foot locator' if geometry.get('model_datums') is not None
                                   else 'robot world XY origin and approximate motor envelope centre'),
            'physical_fit_or_route_acceptance': False}


def validate_shared_station_scenes(station, geometry, boxes, docs):
    """Also prove all model scenes retain one camera and one table geometry."""
    _require(set(docs) == set(FIXTURES), 'All three scenes are required for a common-camera/table proof')
    reference = docs[FIXTURES[0]]
    reference_camera = reference['camera_T_base_link']
    reference_table = _one_obstacle(reference, 'table')
    results = {}
    for fid in FIXTURES:
        doc = docs[fid]
        _require(doc['terminal_fixture'] == fid, 'Scene/model key mismatch')
        results[fid] = validate_station_scene(station, geometry, boxes, doc, world_T_camera=reference_camera)
        table = _one_obstacle(doc, 'table')
        for key in ('center_xy', 'size_xy', 'z_from_floor'):
            _same(table[key], reference_table[key], fid + ': per-model table geometry is prohibited')
        _require(doc.get('camera_keepout') == reference.get('camera_keepout'), fid + ': per-model camera keepout is prohibited')
        camera_body = _one_obstacle(doc, 'hik_camera')
        reference_body = _one_obstacle(reference, 'hik_camera')
        for key in ('T', 'size'):
            _same(camera_body[key], reference_body[key], fid + ': per-model camera body is prohibited')
    return {'passed': True, 'fixtures': results, 'common_camera_and_table': True,
            'camera_visibility_verified': False}
