import unittest
import numpy as np
from momart_bbox_memory.detector import Detection
from .painting import draw_bbox, OnlineBoxedRGB

class PaintingTests(unittest.TestCase):
    def test_outline_only_and_source_unchanged(self):
        image=np.full((10,20,3),70,np.uint8)
        result=draw_bbox(image,[.25,.2,.75,.8],True)
        np.testing.assert_array_equal(image,np.full_like(image,70))
        np.testing.assert_array_equal(result[2,5:15],np.tile([255,0,0],(10,1)))
        np.testing.assert_array_equal(result[3:7,6:14],image[3:7,6:14])
        np.testing.assert_array_equal(result[:2],image[:2])

    def test_missing_keeps_image(self):
        image=np.arange(300,dtype=np.uint8).reshape(10,10,3)
        np.testing.assert_array_equal(draw_bbox(image,[0,0,0,0],False),image)

    def test_train_online_parity_all_recent_frames(self):
        rng=np.random.default_rng(8)
        rgb=rng.integers(0,256,(2,2,10,20,3),dtype=np.uint8)
        ds=[Detection(np.array([.2,.2,.8,.8]),.8,v,False,1)
            for v in [True,False,True,True]]
        class Detector:
            def __init__(self): self.offset=0
            def __call__(self,images):
                start=self.offset;self.offset+=len(images)
                return ds[start:self.offset]
        original=np.moveaxis(rgb,-1,2).astype(np.float32)/255
        wrist=np.zeros_like(original)
        obs={"rgb":original.copy(),"rgb_wrist":wrist,"proprio":np.zeros((2,2,10))}
        painted=OnlineBoxedRGB(Detector(),batch_size=3)(obs)
        expected=np.stack([draw_bbox(im,d.box,d.valid)
            for im,d in zip(rgb.reshape(-1,10,20,3),ds)]).reshape(rgb.shape)
        np.testing.assert_allclose(painted["rgb"],np.moveaxis(expected,-1,2)/255,atol=1e-7)
        np.testing.assert_array_equal(obs["rgb"],original)
        self.assertIs(painted["rgb_wrist"],wrist)
        self.assertIs(painted["proprio"],obs["proprio"])

if __name__=="__main__": unittest.main()
