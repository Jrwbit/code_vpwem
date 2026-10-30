import unittest

import numpy as np

from .interaction import (
    InteractionFrame,
    annotation_hit_rate,
    mask_to_box,
    propagate_keyframes,
    transform_box_to_crop,
)


class InteractionAnnotationTests(unittest.TestCase):
    def test_mask_to_tight_box_and_missing_mask(self):
        mask = np.zeros((10, 20), bool)
        mask[2:6, 4:9] = True
        np.testing.assert_allclose(mask_to_box(mask), (.2, .2, .45, .6))
        self.assertIsNone(mask_to_box(np.zeros_like(mask)))

    def test_propagation_requires_identity_and_preserves_occlusion(self):
        keyframes = {
            0: InteractionFrame("bowl", (.1, .2, .3, .4)),
            2: InteractionFrame("bowl", (.3, .2, .5, .4)),
            4: InteractionFrame("bowl", None, None, False, True),
            6: InteractionFrame("cup", (.6, .2, .8, .4)),
        }
        rows = propagate_keyframes(keyframes, 7, max_gap=3)
        self.assertEqual(rows[1].target_box, (.2, .2, .4, .4))
        self.assertTrue(rows[3].occluded)
        self.assertIsNone(rows[3].target_box)
        self.assertIsNone(rows[5].target_id)  # identity changed across the gap

    def test_crop_transform_is_normalized_in_crop_coordinates(self):
        box = transform_box_to_crop((.25, .25, .75, .75), (25, 10, 75, 90), (100, 100))
        np.testing.assert_allclose(box, (.0, 15 / 80, 1., 65 / 80))
        self.assertIsNone(transform_box_to_crop((.0, .0, .1, .1), (50, 50, 100, 100), (100, 100)))

    def test_hit_rate_separates_missing_and_false_positive(self):
        result = annotation_hit_rate(
            [(0, 0, .5, .5), None, (.1, .1, .2, .2)],
            [(0, 0, .5, .5), (.1, .1, .2, .2), None],
        )
        self.assertEqual(result["visible_frames"], 2)
        self.assertEqual(result["iou_hits"], 1)
        self.assertEqual(result["missing_rate"], .5)
        self.assertEqual(result["false_positive"], 1)


if __name__ == "__main__":
    unittest.main()
