import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr
from .annotate_bbox import annotate
from .detector import Detection

class FakeDetector:
    metadata={"test":"fixture"}
    calls=0
    def __call__(self,images):
        self.calls+=1
        return [Detection(np.array([.1,.2,.8,.9],np.float32),.9,True,False,1) for _ in images]

class AnnotationTests(unittest.TestCase):
    def test_complete_resume_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"source.zarr"; dst=Path(tmp)/"bbox.zarr"
            root=zarr.open(str(src),mode="w")
            root.create_dataset("data/rgb",data=np.zeros((5,9,12,3),np.uint8))
            root.create_dataset("meta/episode_ends",data=np.array([2,5]))
            detector=FakeDetector()
            annotate(src,dst,detector,2)
            result=zarr.open(str(dst),mode="r")
            self.assertTrue(result.attrs["complete"])
            self.assertEqual(result.attrs["completed_frames"],5)
            self.assertEqual(detector.calls,3)
            np.testing.assert_allclose(result["boxes"][4],[.1,.2,.8,.9])
            annotate(src,dst,detector,2)
            self.assertEqual(detector.calls,3)
            detector.metadata={"test":"changed"}
            with self.assertRaises(ValueError):
                annotate(src,dst,detector,2)
            self.assertEqual(list(root["data"].keys()),["rgb"])

if __name__=="__main__":
    unittest.main()
