import copy
import unittest

import numpy as np

from search_fixture_layout import FIXTURES, flange_target, relocate


class LayoutTests(unittest.TestCase):
    def test_rigid_relocation_preserves_camera_motor_and_table_relationship(self):
        box = np.eye(4); box[:3, 3] = [.4, 0, .8]
        camera = np.eye(4); camera[:3, 3] = [.4, 0, 1.3]
        motor = np.eye(4); motor[:3, 3] = [.4, 0, .6]
        scene = {'box_T_base_link':box.tolist(), 'camera_T_base_link':camera.tolist(),
                 'floor_z':-.2, 'obstacles':[{'name':'motor_body', 'T':motor.tolist()},
                 {'name':'table','center_xy':[.5,0], 'size_xy':[.8,1.], 'z_from_floor':[.7,.76]}]}
        original = copy.deepcopy(scene)
        moved, moved_camera = relocate(scene,[.2,.6,.9],90)
        new_box = np.array(moved['box_T_base_link'])
        np.testing.assert_allclose(np.linalg.inv(new_box)@moved_camera,np.linalg.inv(box)@camera,atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(new_box)@moved['obstacles'][0]['T'],np.linalg.inv(box)@motor,atol=1e-12)
        table = moved['obstacles'][1]
        expected = np.linalg.inv(box) @ [.5,0,.53,1]
        np.testing.assert_allclose(np.linalg.inv(new_box)@np.array(table['T'])[:,3],expected,atol=1e-12)
        self.assertEqual(table['size'][:2],[.8,1.])
        self.assertAlmostEqual(table['size'][2],.06)
        self.assertEqual(scene,original)

    def test_each_fixture_center_reaches_same_box_and_retreats_along_box_normal(self):
        box = np.eye(4); box[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
        box[:3,3] = [.2,.6,.9]
        site={'tcp_T_objects':{}, 'flange_T_tcp':np.eye(4).tolist(), 'fixture_z_adjustments_mm':{'JL132':-22}}
        for i, fixture in enumerate(FIXTURES):
            offset=np.eye(4);offset[:3,3]=[.04*i,.02,.03]
            site['tcp_T_objects'][fixture]=offset.tolist()
            target=flange_target(box,site,fixture,.11)
            offset[2,3] += site['fixture_z_adjustments_mm'].get(fixture,0)/1000
            recovered=target@offset
            np.testing.assert_allclose(recovered[:3,3],box[:3,3]-.11*box[:3,2],atol=1e-12)


if __name__=='__main__':
    unittest.main()
