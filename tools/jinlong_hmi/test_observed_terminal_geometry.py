#!/usr/bin/env python3
"""Regression tests for measured frame separation and scan solid preservation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_observed_terminal_geometry as geometry


def minimal_original_builder(scene):
    import mujoco
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name='camera_keepout', type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[0, 0, 2], size=[.35, .35, .4])
    return spec.compile()


class ObservedGeometryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evidence = self.root / 'independent-observations.json'
        evidence.write_text('{"source": "synthetic test observation, not a tool"}\n')
        self.evidence = {'path': str(evidence), 'sha256': geometry.sha256(evidence)}
        self.source = {'model': 'JL100', 'units': 'metres', 'parts': [],
                       'boxes': [{'center': [.02, .03, .04], 'half_size': [.01, .02, .03],
                                  'unknown': True}], 'unknown_columns': 13}
        source_path = self.root / 'source.json'
        self.write_source(source_path)
        self.transform = np.array([[0., -1, 0, .1], [1, 0, 0, .2], [0, 0, 1, -.01], [0, 0, 0, 1]])
        self.station = np.array([[0., 1, 0, .05], [1, 0, 0, 0], [0, 0, -1, .244], [0, 0, 0, 1]])
        self.spec = {
            'observed_frame': {'id': 'capture/observed', 'definition': '+X independent cable side, +Z into box'},
            'source_geometry': {'path': str(source_path), 'sha256': geometry.sha256(source_path),
                                'frame_id': 'CAD/box', 'kind': 'cad'},
            'geometry_registration': {'status': 'verified', 'source_frame_id': 'CAD/box',
                                      'observed_frame_id': 'capture/observed',
                                      'observed_T_source': self.transform.tolist(),
                                      'method': 'Independent synthetic matched features', 'evidence': [self.evidence]},
            'station_registration': {'status': 'verified', 'station_frame_id': 'common-foot-datum',
                                     'observed_frame_id': 'capture/observed',
                                     'station_T_observed': self.station.tolist(),
                                     'method': 'Independent synthetic foot datum', 'evidence': [self.evidence]},
            'observation': {'id': 'capture', 'frame_id': 'capture/observed', 'basis': 'rgbd',
                            'independent_of_tool': True, 'evidence': [self.evidence]},
            'terminals': [self.feature(i) for i in range(7)],
        }

    def feature(self, index):
        return {'id': f'feature-{index}', 'kind': 'ground' if index == 6 else 'stud',
                'role': 'mating' if index < 3 else 'other' if index < 6 else 'ground',
                'collision_name': ('term_JL100_gnd' if index == 6 else
                                   f'term_JL100_stud_{index + 1}' if index < 3 else f'term_JL100_other_{index - 2}'),
                'position_m': [.01 * index, .02, .008], 'axis': [0, 0, 1],
                'radius_m': .004 if index == 6 else .003,
                'length_m': .008 if index == 6 else .03,
                'depth_status': 'measured', 'dimension_source': 'Synthetic known cylinder'}

    def write_source(self, path=None):
        path = path or Path(self.spec['source_geometry']['path'])
        path.write_text(json.dumps(self.source))
        if hasattr(self, 'spec'):
            self.spec['source_geometry']['sha256'] = geometry.sha256(path)

    def write_input(self, models=None):
        path = self.root / 'input.json'
        path.write_text(json.dumps({'schema_version': geometry.SCHEMA_VERSION,
                                    'models': models or {'JL100': self.spec}}))
        return path

    def prepare(self):
        return geometry.prepare_model('JL100', self.spec, self.root)

    def test_requires_explicit_source_and_station_registrations(self):
        for key in ['geometry_registration', 'station_registration']:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.spec)
                bad.pop(key)
                with self.assertRaisesRegex(geometry.GeometryInputError, key):
                    geometry.prepare_model('JL100', bad, self.root)

    def test_rejects_implicit_frame_alias_and_reflection(self):
        self.spec['source_geometry']['frame_id'] = 'capture/observed'
        with self.assertRaisesRegex(geometry.GeometryInputError, 'must remain distinct'):
            self.prepare()
        self.spec['source_geometry']['frame_id'] = 'CAD/box'
        self.spec['geometry_registration']['observed_T_source'] = np.diag([-1., 1, 1, 1]).tolist()
        with self.assertRaisesRegex(geometry.GeometryInputError, 'no reflection'):
            self.prepare()

    def test_rejects_tool_generated_observations_and_missing_stud(self):
        for edit in [lambda spec: spec['observation'].update(independent_of_tool=False),
                     lambda spec: spec['observation'].update(basis='fixture_inverse'),
                     lambda spec: spec['terminals'].pop()]:
            with self.subTest(edit=edit):
                bad = copy.deepcopy(self.spec)
                edit(bad)
                with self.assertRaises(geometry.GeometryInputError):
                    geometry.prepare_model('JL100', bad, self.root)

    def test_collision_numbering_is_explicit_and_not_photo_array_order(self):
        self.spec['terminals'][0]['collision_name'] = 'term_JL100_stud_3'
        self.spec['terminals'][2]['collision_name'] = 'term_JL100_stud_1'
        self.spec['terminals'].reverse()
        doc = self.prepare()
        by_name = {x['name']: x for x in doc['measured_terminals']}
        self.assertEqual(by_name['term_JL100_stud_1']['center_source'], 'feature-2')
        self.assertEqual(by_name['term_JL100_stud_3']['center_source'], 'feature-0')
        self.spec['terminals'][0].pop('collision_name')
        with self.assertRaises(geometry.GeometryInputError):
            self.prepare()

    def test_rigid_rotation_retains_unknown_box_volume(self):
        doc = self.prepare()
        part = doc['parts'][0]
        self.assertTrue(part['unknown'])
        self.assertEqual(doc['preservation']['source_unknown_columns'], 13)
        source_box = self.source['boxes'][0]
        vertices = np.asarray(part['vertices'])
        recovered = geometry.transform_points(vertices, np.linalg.inv(self.transform))
        np.testing.assert_allclose(recovered.min(0), np.array(source_box['center']) - source_box['half_size'], atol=1e-14)
        np.testing.assert_allclose(recovered.max(0), np.array(source_box['center']) + source_box['half_size'], atol=1e-14)
        self.assertEqual(len(doc['measured_terminals']), 7)
        self.assertEqual({t['name'] for t in doc['measured_terminals']},
                         {f'term_JL100_stud_{i}' for i in range(1, 4)} |
                         {f'term_JL100_other_{i}' for i in range(1, 4)} | {'term_JL100_gnd'})

    def test_scan_terminal_holes_cannot_be_reused_with_shifted_template(self):
        original = geometry.terminal_geometry('JL100', self.spec)
        self.source['measured_terminals'] = copy.deepcopy(original)
        self.spec['source_geometry']['kind'] = 'scan'
        for feature, terminal in zip(self.spec['terminals'], original):
            feature['source_terminal_name'] = terminal['name']
            feature['position_m'] = geometry.transform_points([terminal['a']], self.transform)[0].tolist()
            feature['axis'] = (self.transform[:3, :3] @ np.array([0, 0, 1])).tolist()
        self.write_source()
        doc = self.prepare()
        self.assertTrue(doc['preservation']['scan_terminal_cylinders_preserved'])
        self.spec['terminals'][0]['position_m'][0] += .0001
        with self.assertRaisesRegex(geometry.GeometryInputError, 'terminal-excluded voxels'):
            self.prepare()

    def test_sha_rejects_changed_evidence(self):
        Path(self.evidence['path']).write_text('changed after registration')
        with self.assertRaisesRegex(geometry.GeometryInputError, 'SHA256 mismatch'):
            self.prepare()

    def test_bundle_with_missing_registration_writes_diagnostics_only(self):
        other = copy.deepcopy(self.spec)
        other.pop('geometry_registration')
        out = self.root / 'out'
        result = geometry.build(self.write_input({'JL100': self.spec, 'JL132': other}), out)
        self.assertEqual(result['status'], 'rejected_incomplete_registration')
        self.assertFalse(list(out.glob('*_collision.json')))
        self.assertTrue((out / 'registration_diagnostics.json').exists())

    def test_scene_uses_observed_transform_and_preserves_other_obstacles(self):
        out = self.root / 'out'
        self.assertFalse(geometry.build(self.write_input(), out)['errors'])
        world = np.eye(4)
        world[:3, 3] = [.43, .05, .55]
        scene = {'box_T_base_link': (world @ self.station).tolist(), 'terminals': True,
                 'world_T_station': world.tolist(), 'tool_cad_collision': '/some/original/tool.json',
                 'obstacles': [{'name': 'box_old'}, {'name': 'camera_keepout', 'size': [.7, .7, .8]},
                               {'name': 'table'}, {'name': 'motor_body'}]}
        original = copy.deepcopy(scene)
        result = geometry.bind_scene(scene, 'JL100', world, out)
        self.assertEqual(scene, original)
        np.testing.assert_allclose(result['box_T_base_link'], world @ self.station)
        self.assertFalse(result['terminals'])
        self.assertEqual(result['obstacles'], original['obstacles'][1:])
        self.assertEqual(result['tool_cad_collision'], original['tool_cad_collision'])
        self.assertIs(geometry.modify_scene(scene, 'JL100', out), scene)
        self.assertEqual(scene, result)
        del scene['world_T_station']
        with self.assertRaisesRegex(geometry.GeometryInputError, 'explicit world_T_station'):
            geometry.modify_scene(scene, 'JL100', out)

    def test_binding_rejects_old_box_frame_instead_of_leaving_stale_targets(self):
        out = self.root / 'out'
        geometry.build(self.write_input(), out)
        scene = {'box_T_base_link': np.eye(4).tolist(),
                 'world_T_station': np.eye(4).tolist(), 'terminals': True, 'obstacles': []}
        original = copy.deepcopy(scene)
        with self.assertRaisesRegex(geometry.GeometryInputError, 'rebuild targets in the observed frame'):
            geometry.modify_scene(scene, 'JL100', out)
        self.assertEqual(scene, original)

    def test_assumed_depth_not_reported_as_measured(self):
        self.spec['terminals'][0]['depth_status'] = 'assumed'
        result = self.prepare()
        self.assertEqual(result['status'], 'offline_registered_orientation_candidate_assumed_depth')
        self.assertIn('cannot certify physical mating', result['limitations'][-1])

    def test_registered_adapter_compiles_all_seven_terminals_and_unknown_solid(self):
        import mujoco
        out = self.root / 'out'
        geometry.build(self.write_input(), out)
        scene = {'world_T_station': np.eye(4).tolist(), 'obstacles': [], 'terminals': True}
        geometry.modify_scene(scene, 'JL100', out)
        model = geometry.build_model(scene, 'JL100', minimal_original_builder)
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
        self.assertEqual(len(names), 9)  # camera, retained unknown box hull, 7 terminals
        self.assertIn('camera_keepout', names)
        self.assertIn('box_JL100_cad_000', names)
        self.assertIn('term_JL100_other_3', names)
        self.assertIn('term_JL100_gnd', names)

    def test_template_packaging_uses_explicit_reverse_row_mapping_and_p5(self):
        feature_ids = ['U1', 'V1', 'W1', 'W2', 'U2', 'V2', 'ground']
        template = {'model': 'JL100', 'tool_geometry_used_for_measurement': False,
                    'source_task': 'capture', 'frame': 'Independent observed frame',
                    'camera_T_box': np.eye(4).tolist(),
                    'features': [{'id': name, 'xyz_box_mm': [i, 2., 9.],
                                  'cap_surface_depth_mm_percentiles_5_50_95': [7., 8., 10.]}
                                 for i, name in enumerate(feature_ids)]}
        path = self.root / 'template.json'
        path.write_text(json.dumps(template))
        packaged = geometry.model_from_template('JL100', path, self.spec['source_geometry'],
                self.spec['geometry_registration'], self.spec['station_registration'],
                ['W1', 'V1', 'U1'], ['V2', 'U2', 'W2'], [self.evidence])
        by_name = {x['collision_name']: x for x in packaged['terminals']}
        self.assertEqual(by_name['term_JL100_stud_1']['id'], 'W1')
        self.assertEqual(by_name['term_JL100_other_3']['id'], 'W2')
        self.assertEqual(by_name['term_JL100_stud_1']['position_m'], [.002, .002, .007])

    def test_no_overwriting_prior_geometry(self):
        out = self.root / 'out'
        source = self.write_input()
        geometry.build(source, out)
        with self.assertRaisesRegex(geometry.GeometryInputError, 'historical evidence'):
            geometry.build(source, out)

    def nominal_station(self):
        self.spec['station_registration'].update(
            status='nominal_cad_translation_observed_orientation',
            physical_registration_confirmed=False, orientation_evidence=[self.evidence],
            translation_evidence=[self.evidence],
            translation_limitations=['Synthetic nominal CAD offset; physical foot-to-box translation unmeasured'])
        self.spec['station_registration'].pop('evidence')

    def test_nominal_station_is_design_candidate_without_physical_registration_claim(self):
        self.nominal_station()
        doc = self.prepare()
        self.assertEqual(doc['status'], 'offline_design_candidate')
        self.assertFalse(doc['physical_motor_registration_confirmed'])
        self.assertIn('nominal CAD', doc['geometry_scope'])
        out = self.root / 'out'
        geometry.build(self.write_input(), out)
        frames = json.loads((out / 'registered_frames.json').read_text())
        self.assertFalse(frames['physical_motor_registration_confirmed'])
        scene = geometry.bind_scene({'obstacles': []}, 'JL100', np.eye(4), out)
        self.assertFalse(scene['physical_motor_registration_confirmed'])

    def test_nominal_station_cannot_claim_confirmed_physical_registration(self):
        self.nominal_station()
        for flag in [None, True]:
            self.spec['station_registration']['physical_registration_confirmed'] = flag
            with self.assertRaisesRegex(geometry.GeometryInputError, 'physical_registration_confirmed=false'):
                self.prepare()

    def test_nominal_station_requires_separate_evidence_and_translation_limits(self):
        self.nominal_station()
        for key in ('orientation_evidence', 'translation_evidence', 'translation_limitations'):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.spec)
                bad['station_registration'].pop(key)
                with self.assertRaises(geometry.GeometryInputError):
                    geometry.prepare_model('JL100', bad, self.root)

    def test_nominal_status_cannot_replace_verified_source_geometry_registration(self):
        self.nominal_station()
        self.spec['geometry_registration']['status'] = 'nominal_cad_translation_observed_orientation'
        with self.assertRaisesRegex(geometry.GeometryInputError, 'geometry_registration: registration is not verified'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
