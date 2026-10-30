import unittest
from temporal_metrics import match,miss_runs,event_scores,wilson


class TemporalContracts(unittest.TestCase):
    def test_same_class_instance_cannot_substitute_target(self):
        truth=[dict(class_id=5,box=[0,0,10,10],instance_id='A'),dict(class_id=5,box=[20,0,30,10],instance_id='B')]
        self.assertEqual(match([dict(class_id=5,box=[20,0,30,10],score=.9)],truth),{0:'B'})

    def test_invisibility_breaks_miss_streak_and_gaps_are_not_frames(self):
        rows=[dict(id=str(t),step=t) for t in [0,1,2,7,8]]
        truth={r['id']:([dict(instance_id='A')] if r['step']!=2 else []) for r in rows}
        hits={r['id']:set() for r in rows}
        dense=miss_runs(rows,truth,hits)
        self.assertEqual(dense['length_in_observed_samples']['maximum'],2)
        self.assertAlmostEqual(dense['span_seconds']['maximum'],.1)

    def test_first_hit_does_not_use_future_or_other_instance(self):
        rows=[dict(id=str(t),step=t,episode_id='e') for t in range(20)]
        truth={r['id']:[dict(instance_id='A')] for r in rows}
        hits={r['id']:({'A'} if r['step']==11 else {'B'}) for r in rows}
        value=event_scores([dict(episode_id='e',instance_id='A',onset=10,end=14)],rows,truth,hits,set(truth))
        self.assertFalse(value['events'][0]['dense']['first_hit'])
        self.assertAlmostEqual(value['events'][0]['dense']['first_post_motion_hit_delay_seconds'],.05)

    def test_wilson_point_estimate_is_not_a_95_percent_lower_bound(self):
        self.assertAlmostEqual(wilson(32,1)[0],.847,places=3)


if __name__=='__main__':unittest.main()
