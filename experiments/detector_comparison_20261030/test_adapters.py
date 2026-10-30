"""Adapter contracts exercised in the actual detector environment."""
import unittest
import numpy as np
from benchmark import sanitize


class AdapterContracts(unittest.TestCase):
    def test_clipping_keeps_nms_dtypes_consistent(self):
        rgb=np.zeros((90,120,3),np.uint8)
        objects=[dict(class_id=1,box=[-1.,-2.,121.,91.],score=.8),
                 dict(class_id=1,box=[0.,0.,120.,90.],score=.7)]
        result=sanitize(rgb,objects)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['box'],[0.,0.,120.,90.])

    def test_red_evidence_gate_does_not_remove_other_categories(self):
        rgb=np.zeros((90,120,3),np.uint8)
        objects=[dict(class_id=c,box=[2.,2.,10.,10.],score=.8) for c in [0,2]]
        self.assertEqual([o['class_id'] for o in sanitize(rgb,objects)],[2])


if __name__=='__main__':unittest.main()
