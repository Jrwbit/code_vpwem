import unittest
import numpy as np
from .detector import Detection, QualityGatedDetector
from .quality import quality_gate
from .evaluate_online import summarize


class OnlineTests(unittest.TestCase):
    def test_gate_cache_parity_and_missing_rgb(self):
        from momart_boxed_no_history.painting import OnlineBoxedRGB
        box = np.array([.1,.1,.8,.8], np.float32)
        ds = [Detection(box,.8,True,False,1), Detection(box,.9,True,True,2),
              Detection(box,.3,True,False,1), Detection(box,0,False,False,0)]
        class Fake:
            metadata = {}
            def __call__(self, images): return ds
        detector = QualityGatedDetector(Fake())
        accepted = detector.filter(ds)
        expected = quality_gate([d.box for d in ds], [d.valid for d in ds],
                                [d.ambiguous for d in ds], [d.score for d in ds])
        np.testing.assert_array_equal([d.valid for d in accepted], expected)
        image = np.full((1,4,3,10,20), .4, np.float32)
        actual = OnlineBoxedRGB(detector)({"rgb": image})["rgb"]
        np.testing.assert_array_equal(actual[:,1:], image[:,1:])
        self.assertFalse(np.array_equal(actual[:,:1], image[:,:1]))

    def test_metrics_missing_and_invisible_denominators(self):
        rows=[]
        for segment, pixels, valid, iou in [(0,5,True,.8),(0,5,False,0),
                                            (1,5,False,0),(1,0,True,0)]:
            rows.append(dict(demo="test",segment=segment,visibility_pixels=pixels,
                             gated_valid=valid,gated_iou=iou))
        result=summarize(rows)
        self.assertEqual(result["hit_rate"],1/3)
        self.assertEqual(result["missing_rate"],2/3)
        self.assertEqual(result["localization_precision"],.5)
        self.assertEqual(result["absent_false_positive_rate"],1)
        self.assertEqual(result["longest_visible_missing_samples"],1)
        self.assertIsNone(summarize([])["hit_rate"])


if __name__ == "__main__": unittest.main()
