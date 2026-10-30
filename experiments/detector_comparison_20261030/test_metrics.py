import unittest
from metrics import measure, calibrate


class MatchingContracts(unittest.TestCase):
    def test_duplicate_and_wrong_class_do_not_inflate_recall(self):
        labels={'x':dict(objects=[dict(class_id=0,box=[0,0,10,10])])}
        predictions=[dict(id='x',objects=[dict(class_id=c,box=[0,0,10,10],score=s)
                                        for c,s in [(0,.9),(0,.8),(1,.95)]])]
        result=measure(predictions,labels,[.25]*6)
        self.assertEqual(result['micro']['tp'],1)
        self.assertEqual(result['micro']['fp'],2)
        self.assertEqual(result['micro']['fn'],0)

    def test_empty_predictions_are_not_perfect_precision(self):
        labels={'x':dict(objects=[dict(class_id=2,box=[0,0,10,10])])}
        result=measure([dict(id='x',objects=[])],labels,[.25]*6)
        self.assertIsNone(result['micro']['precision'])
        self.assertEqual(result['micro']['recall'],0.)

    def test_one_correct_prediction_does_not_certify_95_percent(self):
        labels={'x':dict(objects=[dict(class_id=0,box=[0,0,10,10])])}
        records=[dict(id='x',objects=[dict(class_id=0,box=[0,0,10,10],score=.8)])]
        result=calibrate(records,labels)
        self.assertFalse(any(result['dev_precision_target_feasible']))

    def test_visible_box_matching_uses_exclusive_pixel_bounds(self):
        labels={'x':dict(objects=[dict(class_id=2,box=[4,5,5,7])])}
        records=[dict(id='x',objects=[dict(class_id=2,box=[4,5,5,7],score=.8)])]
        self.assertEqual(measure(records,labels,[.25]*6,.75)['micro']['tp'],1)


if __name__=='__main__':unittest.main()
