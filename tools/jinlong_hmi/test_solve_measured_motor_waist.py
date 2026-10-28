"""Measured inputs must not silently reuse a design certificate or move a camera."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from solve_measured_motor_waist import connect_rrt, load_observations, measured_scene, rigid


class MeasuredSceneTests(unittest.TestCase):
    def test_only_box_attached_objects_follow_measurement(self):
        pose = np.eye(4).tolist()
        original = {
            "terminal_fixture": "JL132", "box_T_base_link": pose,
            "obstacles": [{"name": n, "T": pose} for n in
                          ("box_wall", "motor_body", "motor_fitting_shaft", "table", "hik_camera", "robot")],
            "shared_motor_station": {"old": True},
            "shared_motor_station_validation": {"passed": True},
        }
        before = copy.deepcopy(original)
        observed = np.eye(4)
        observed[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        observed[:3, 3] = [.4, -.1, .8]
        result, target = measured_scene(original, observed, np.eye(4), 40)
        self.assertEqual(original, before)
        for ob in result["obstacles"]:
            expected = observed if ob["name"] in ("box_wall", "motor_body", "motor_fitting_shaft") else np.eye(4)
            np.testing.assert_allclose(ob["T"], expected)
        np.testing.assert_allclose(target[:3, 3], [.4, -.1, .76])
        self.assertNotIn("shared_motor_station_validation", result)
        self.assertFalse(result["physical_motor_registration_confirmed"])
        self.assertTrue(result["nominal_station_provenance"]["old"])

    def test_reject_nonrigid_pose(self):
        for bad in (np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                rigid(bad)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.camera = np.eye(4)
        self.camera[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.camera[0, 3] = .4
        self.document = {"world_T_camera": self.camera.tolist(), "models": {}}
        for model in ("JM160", "JL132", "JL100"):
            box = np.eye(4)
            box[0, 3] = .1
            path = self.root / (model + ".json")
            path.write_text(json.dumps({"success": True, "model_type": model[2:], "camera_T_box": box.tolist()}))
            self.document["models"][model] = {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def load(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.document))
        return load_observations(path)

    def change_capture(self, model, key, value, update_hash=True):
        path = self.root / (model + ".json")
        record = json.loads(path.read_text())
        record[key] = value
        path.write_text(json.dumps(record))
        if update_hash:
            self.document["models"][model]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_camera_composition_order(self):
        _, boxes, sources = self.load()
        np.testing.assert_allclose(boxes["JL100"][:3, 3], [.4, .1, 0])
        self.assertEqual(len(sources), 3)

    def test_failed_detection_cannot_be_a_motion_target(self):
        self.change_capture("JM160", "success", False)
        with self.assertRaisesRegex(ValueError, "capture failed"):
            self.load()

    def test_mislabeled_or_modified_capture_is_rejected(self):
        self.change_capture("JL132", "model_type", "160")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load()
        self.change_capture("JL132", "model_type", "132", update_hash=False)
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.load()

    def test_all_three_models_required(self):
        self.document["models"].pop("JL100")
        with self.assertRaisesRegex(ValueError, "three"):
            self.load()


class ConnectionTests(unittest.TestCase):
    def test_rrt_adapter_keeps_waist_and_preserves_dense_policy(self):
        limits = {f"AR5-5_07L_joint_{i}": (-2., 2.) for i in range(1, 8)}
        ctx = SimpleNamespace(robot=SimpleNamespace(lim=limits), fixture="JL132")
        calls = []
        validator = SimpleNamespace(
            ClearancePolicy=lambda **kw: kw or {"dense": True},
            check_segment=lambda *a, **kw: calls.append((a, kw)) or {"ok": True})
        a = np.r_[[.1, .2, .3, .4], np.zeros(7)]
        b = a.copy(); b[4:] = .5
        def planner(start, end, lo, hi, check, rng, **kw):
            check(start, end, True)
            return {"ok": True, "arm_rows": [start, end]}
        with patch("solve_measured_motor_waist.rrt_connect", planner):
            result = connect_rrt(ctx, a, b, None, validator,
                                 seconds=1, n_samples=1, seed_arm_rows=[])
        np.testing.assert_allclose(result["rows"], [a, b])
        self.assertEqual(calls[0][0][5], {"dense": True})
        np.testing.assert_allclose(calls[0][1]["waist"], a[:4])

    def test_rrt_adapter_rejects_changing_waist(self):
        with self.assertRaisesRegex(ValueError, "one waist"):
            connect_rrt(None, np.zeros(11), np.ones(11), None, None,
                        seconds=1, n_samples=1, seed_arm_rows=[])


if __name__ == "__main__":
    unittest.main()
