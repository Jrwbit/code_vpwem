import unittest
import numpy as np
from .collect_online_corpus import split_episodes
from .cache_strategy_proposals import restore_crop
from .strategies import CausalTracker,Config


class StrategyTests(unittest.TestCase):
    def test_disjoint_splits_exclude_old_test(self):
        s=split_episodes([f"t{i}" for i in range(10)],[f"v{i}" for i in range(8)],["v0","v1","v2"])
        self.assertFalse(set(s["fit"])&set(s["dev"]))
        self.assertFalse(set(s["test"])&{"v0","v1","v2"})
        self.assertEqual(sum(map(len,s.values())),8)

    def test_local_crop_returns_original_coordinates(self):
        boxes=restore_crop([[0,0,20,10]],(30,20,50,30),(90,120,3))
        np.testing.assert_allclose(boxes,[[.25,20/90,50/120,30/90]])

    def test_low_score_requires_recent_matching_identity(self):
        tracker=CausalTracker(Config(low_threshold=.15))
        image=np.zeros((90,120,3),np.uint8)
        a=np.array([[.2,.2,.4,.4]],np.float32)
        self.assertFalse(tracker.step(image,a,[.2])[0].valid)
        self.assertTrue(tracker.step(image,a,[.8])[0].valid)
        self.assertTrue(tracker.step(image,a,[.2])[0].valid)
        far=np.array([[.8,.8,.9,.9]],np.float32)
        self.assertFalse(tracker.step(image,far,[.2])[0].valid)
        tracker.reset()
        self.assertFalse(tracker.step(image,a,[.2])[0].valid)

    def test_no_evidence_does_not_propagate_stale_boxes(self):
        tracker=CausalTracker(Config(method="template"))
        image=np.zeros((90,120,3),np.uint8)
        tracker.step(image,[[.2,.2,.4,.4]],[.8])
        for _ in range(6):
            d,_=tracker.step(image,[],[])
            self.assertFalse(d.valid)
        self.assertIsNone(tracker.box)

    def test_fallback_keeps_a_reliable_existing_box(self):
        from .compare_strategies import predict_stream
        image=np.zeros((90,120,3),np.uint8)
        original=np.array([.2,.2,.5,.5],np.float32)
        prop=dict(boxes=[original],scores=[.8],roi_boxes=[],roi_scores=[])
        probs=np.zeros((1,90,120),np.float32);probs[:,25:40,30:45]=.99
        spec=dict(method="hybrid",threshold=.25,mask_threshold=.5,
                  confidence=.9,refine_existing=False)
        detected,_=predict_stream(spec,[image],[prop],probs)[0]
        np.testing.assert_array_equal(detected.box,original)

    def test_runtime_streams_do_not_share_tracks(self):
        from .precision_detector import PrecisionDetector
        runtime=PrecisionDetector.__new__(PrecisionDetector)
        runtime.spec=dict(method="track",track=dict(method="association",low_threshold=.15))
        runtime.tiny=None;runtime.states={}
        class Fake:
            def proposals(self,images,threshold):
                score=.8 if images[0][0,0,0]==1 else .2
                return [dict(boxes=np.array([[24,18,48,36]],np.float32),scores=np.array([score],np.float32))]
        runtime.global_detector=Fake()
        high=np.zeros((90,120,3),np.uint8);high[0,0,0]=1
        low=np.zeros_like(high)
        self.assertTrue(runtime.detect(high,0).valid)
        self.assertFalse(runtime.detect(low,1).valid)
        self.assertTrue(runtime.detect(low,0).valid)
        runtime.reset(0)
        self.assertFalse(runtime.detect(low,0).valid)


if __name__=="__main__":unittest.main()
