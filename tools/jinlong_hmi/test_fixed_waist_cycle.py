"""Focused checks for the collision-reporting optimization used by the study."""
import unittest

import mujoco
import numpy as np

from solve_fixed_waist_cycle import bounded_contact_horizon


def pair_model(other_name, other_type, gap):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    body.add_freejoint()
    tool = body.add_geom(name="tool_pin", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         size=[.01, 0, 0])
    tool.contype, tool.conaffinity, tool.margin = 4, 0, .06
    other = spec.worldbody.add_geom(name=other_name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=[.01, 0, 0], pos=[.02 + gap, 0, 0])
    other.contype, other.conaffinity, other.margin = other_type, 4, .03
    return spec.compile()


def contacts(model):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    return np.array(data.contact.dist[:data.ncon])


class ContactHorizonTest(unittest.TestCase):
    def test_keep_environment_collision_and_solids(self):
        model = pair_model("box_wall", 2, .007)
        before_size, before_position = model.geom_size.copy(), model.geom_pos.copy()
        original = contacts(model)
        proof = bounded_contact_horizon(model)
        np.testing.assert_allclose(contacts(model), original, atol=1e-12)
        np.testing.assert_array_equal(model.geom_size, before_size)
        np.testing.assert_array_equal(model.geom_pos, before_position)
        self.assertGreaterEqual(proof["minimum_coverage_reserve_mm"], .4999)

    def test_keep_generation_buffer(self):
        model = pair_model("box_wall", 2, .0098)
        bounded_contact_horizon(model)
        self.assertEqual(len(contacts(model)), 1)

    def test_keep_existing_twenty_mm_body_padding(self):
        model = pair_model("torso", 1, .027)
        bounded_contact_horizon(model)
        found = contacts(model)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0] - .020, .007, places=10)

    def test_remove_only_distant_environment_contacts(self):
        model = pair_model("box_wall", 2, .040)
        self.assertEqual(len(contacts(model)), 1)
        proof = bounded_contact_horizon(model)
        self.assertEqual(len(contacts(model)), 0)
        self.assertGreaterEqual(proof["unreported_effective_clearance_lower_bound_mm"], 10.49)


if __name__ == "__main__":
    unittest.main()
