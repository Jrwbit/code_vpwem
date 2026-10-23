"""Adversarial regressions for shared motor placement, without simulator/CAD."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_station_constraints import (FIXTURES, validate_shared_station_scenes,
                                        validate_station_geometry, validate_station_scene)


def transform(xyz=(0, 0, 0), yaw=0, down=False):
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    if down:
        T[:3, :3] = T[:3, :3] @ np.diag([1, -1, -1])
    T[:3, 3] = xyz
    return T


def fixture_data(station=None):
    # Different intrinsic heights/yaws are allowed; search-time corrections
    # are not. Four feet remain at one datum despite different motor sizes.
    geometry = {'station_T_boxes': {}, 'motor_envelopes': {}, 'model_datums': {},
                'intrinsic_geometry_confirmed': True}
    for fid, height, yaw in zip(FIXTURES, (.39, .31, .24), (0, 20, 90)):
        geometry['station_T_boxes'][fid] = transform([.055, 0, height], yaw, down=True).tolist()
        geometry['motor_envelopes'][fid] = {'T': transform([.02, 0, .1]).tolist(), 'size': [.35, .25, .2]}
        geometry['model_datums'][fid] = {'foot_datum_z_m': 0, 'shaft_direction_station': [1, 0, 0],
                                       'foot_hole_centers_station_m': [[x, y, 0] for x in (-.1, .1) for y in (-.08, .08)]}
    station = transform([.43, .05, .55]) if station is None else station
    boxes = {f: station @ np.asarray(geometry['station_T_boxes'][f]) for f in FIXTURES}
    camera = transform([.48, .08, 1.4], down=True)
    docs = {}
    for fid in FIXTURES:
        envelope = geometry['motor_envelopes'][fid]
        docs[fid] = {'terminal_fixture': fid, 'box_T_base_link': boxes[fid].tolist(), 'floor_z': -.261,
                     'camera_T_base_link': camera.tolist(), 'camera_keepout': {'center_xy': [.48, .08], 'half_xy': [.35, .35]},
                     'obstacles': [{'name': 'motor_body', 'T': (station @ envelope['T']).tolist(), 'size': envelope['size']},
                                   {'name': 'table', 'center_xy': station[:2, 3].tolist(), 'size_xy': [1, 1],
                                    'z_from_floor': [station[2, 3] + .261 - .06, station[2, 3] + .261]},
                                   {'name': 'hik_camera', 'T': camera.tolist(), 'size': [.27, .09, .08]}],
                     'shared_motor_station': {'world_T_station': station.tolist(),
                                              'station_T_box': geometry['station_T_boxes'][fid],
                                              'tabletop_world_z_m': station[2, 3],
                                              'no_model_specific_placement_or_height_adjustment': True}}
    return station, geometry, boxes, docs


class SharedStationConstraintTests(unittest.TestCase):
    def test_one_station_accepts_intrinsic_model_height_and_yaw_differences(self):
        result = validate_shared_station_scenes(*fixture_data())
        self.assertTrue(result['passed'])
        self.assertFalse(result['camera_visibility_verified'])
        self.assertEqual(len(result['fixtures']), 3)

    def test_single_motor_xyz_or_yaw_adjustment_cannot_make_a_solution(self):
        for offset, yaw in [([.001, 0, 0], 0), ([0, .001, 0], 0), ([0, 0, .001], 0), ([0, 0, 0], .1)]:
            with self.subTest(offset=offset, yaw=yaw):
                station, geometry, boxes, docs = fixture_data()
                boxes['JL100'] = boxes['JL100'] @ transform(offset, yaw)
                # Also falsify the scene and its metadata; the independent
                # intrinsic source still prevents this old free-box strategy.
                docs['JL100']['box_T_base_link'] = boxes['JL100'].tolist()
                docs['JL100']['shared_motor_station']['station_T_box'] = (np.linalg.inv(station) @ boxes['JL100']).tolist()
                with self.assertRaisesRegex(ValueError, 'independent box placement'):
                    validate_shared_station_scenes(station, geometry, boxes, docs)

    def test_whole_station_can_move_and_yaw_once_for_all_models(self):
        result = validate_shared_station_scenes(*fixture_data(transform([.52, .12, .62], 35)))
        self.assertTrue(result['passed'])
        self.assertTrue(all(r['far_side_offset_mm'] > 0 for r in result['fixtures'].values()))

    def test_station_roll_and_nonrigid_matrix_are_rejected(self):
        station, geometry, boxes, docs = fixture_data()
        roll = np.eye(4); roll[1:3, 1:3] = [[math.cos(.1), -math.sin(.1)], [math.sin(.1), math.cos(.1)]]
        for bad in (station @ roll, station @ np.diag([1.01, 1, 1, 1])):
            with self.subTest(station=bad.tolist()):
                with self.assertRaisesRegex(ValueError, 'level|rigid'):
                    validate_station_scene(bad, geometry, boxes, docs['JM160'])

    def test_same_table_cannot_hide_a_model_specific_motor_riser(self):
        for change in ('move_motor', 'float_intrinsic_motor', 'table_height', 'support'):
            with self.subTest(change=change):
                station, geometry, boxes, docs = fixture_data()
                doc = docs['JL100']
                if change == 'move_motor':
                    doc['obstacles'][0]['T'][2][3] += .01
                elif change == 'float_intrinsic_motor':
                    geometry['motor_envelopes']['JL100']['T'][2][3] += .01
                elif change == 'table_height':
                    doc['obstacles'][1]['z_from_floor'][1] += .01
                else:
                    doc['obstacles'].append({'name': 'motor_support_JL100', 'size': [.1, .1, .01]})
                with self.assertRaises(ValueError):
                    validate_shared_station_scenes(station, geometry, boxes, docs)

    def test_camera_or_table_changes_between_models_are_rejected(self):
        for change in ('camera', 'camera_body', 'keepout', 'table_size'):
            with self.subTest(change=change):
                station, geometry, boxes, docs = fixture_data()
                doc = docs['JL132']
                if change == 'camera':
                    doc['camera_T_base_link'][0][3] += .02
                elif change == 'camera_body':
                    doc['obstacles'][2]['T'][0][3] += .02
                elif change == 'keepout':
                    doc['camera_keepout']['half_xy'][0] = .1
                else:
                    doc['obstacles'][1]['size_xy'][0] = .5
                with self.assertRaises(ValueError):
                    validate_shared_station_scenes(station, geometry, boxes, docs)

    def test_motor_fitting_cannot_be_omitted_moved_or_shrunk(self):
        for change in ('none', 'missing', 'move', 'shrink'):
            with self.subTest(change=change):
                station, geometry, boxes, docs = fixture_data()
                geometry['motor_aux_envelopes'] = {f: [] for f in FIXTURES}
                part = {'name': 'motor_fitting_JL100_1', 'T': transform([.06, .1, .18]).tolist(), 'size': [.03, .02, .03]}
                geometry['motor_aux_envelopes']['JL100'] = [part]
                placed = {'name': part['name'], 'T': (station @ part['T']).tolist(), 'size': list(part['size'])}
                if change != 'missing':
                    docs['JL100']['obstacles'].append(placed)
                if change == 'move':
                    placed['T'][1][3] += .005
                elif change == 'shrink':
                    placed['size'][1] -= .005
                if change == 'none':
                    self.assertTrue(validate_shared_station_scenes(station, geometry, boxes, docs)['passed'])
                else:
                    with self.assertRaises(ValueError):
                        validate_shared_station_scenes(station, geometry, boxes, docs)

    def test_box_must_be_on_far_side_of_shared_foot_locator(self):
        station, geometry, boxes, docs = fixture_data(transform([.43, .05, .55], 180))
        with self.assertRaisesRegex(ValueError, 'far side'):
            validate_shared_station_scenes(station, geometry, boxes, docs)

    def test_confirmed_geometry_requires_horizontal_common_shafts_and_foot_datum(self):
        for change in ('missing', 'vertical_shaft', 'opposite_shaft', 'lifted_foot', 'shifted_locator'):
            with self.subTest(change=change):
                _, geometry, _, _ = fixture_data()
                if change == 'missing':
                    geometry.pop('model_datums')
                else:
                    datum = geometry['model_datums']['JL100']
                    if change == 'vertical_shaft':
                        datum['shaft_direction_station'] = [0, 0, 1]
                    elif change == 'opposite_shaft':
                        datum['shaft_direction_station'] = [-1, 0, 0]
                    elif change == 'lifted_foot':
                        datum['foot_datum_z_m'] = .01
                    else:
                        for p in datum['foot_hole_centers_station_m']:
                            p[0] += .01
                with self.assertRaises(ValueError):
                    validate_station_geometry(geometry)


if __name__ == '__main__':
    unittest.main()
