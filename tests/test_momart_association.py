"""Identity regressions for same-class crossings, occlusion and global matching."""
import numpy as np
import pytest
from momart_object_memory.association import AssociationConfig, assign
from momart_object_memory.state import ObjectBank, ObjectObservation, iou


def obs(x, feature=(1., 0.), cls=5):
    return ObjectObservation(cls, np.array([x, .1, x+.2, .3], np.float32),
                             .9, np.array(feature, np.float32))


def bank(**kwargs):
    return ObjectBank(8, 3, 2, overflow='evict_oldest_invisible',
                      association=dict(method='appearance_motion', **kwargs))


def test_same_class_crossing_preserves_histories_and_detection_order():
    for reverse in (False, True):
        b=bank(max_center_distance=.5)
        b.update(0, 0., [obs(.1), obs(.4, (0., 1.))])
        detections=[obs(.35), obs(.15, (0., 1.))]
        b.update(1, .05, detections[::-1] if reverse else detections)
        assert len(b.tracks)==2
        np.testing.assert_array_equal(b.snapshot('a3')['features'][:2, -2:],
                                      [[[1,0],[1,0]], [[0,1],[0,1]]])
        np.testing.assert_allclose(b.tracks[0]['box'][0], .35)


def test_short_occlusion_reassociates_without_fabricating_current_box():
    b=bank();b.update(0,0.,[obs(.1)]);b.update(7,.35,[])
    assert not b.snapshot('a3')['boxes'].any()
    b.update(8,.4,[obs(.2)])
    assert len(b.tracks)==1 and len(b.tracks[0]['events'])==2


def test_stale_or_distant_or_conflicting_appearance_cannot_pollute_history():
    for time, detection in ((1.,obs(.1)), (.1,obs(.7)), (.1,obs(.1,(-1,0))),
                            (.1,obs(.1,cls=4))):
        b=bank();b.update(0,0.,[obs(.1)]);b.update(20,time,[detection])
        assert len(b.tracks)==2
        assert len(b.tracks[0]['events'])==1
        assert not b.tracks[0]['visible']


def test_zero_roi_features_require_iou_instead_of_arbitrary_reidentification():
    b=bank();b.update(0,0.,[obs(.1,(0,0))]);b.update(1,.05,[obs(.4,(0,0))])
    assert len(b.tracks)==2


def test_global_assignment_avoids_greedy_dead_end():
    tracks=[dict(class_id=5, box=np.array([i,0,i+1,1]),events=[(0,0,np.ones(2))]) for i in (0,1)]
    detections=[obs(.1),obs(.2)]
    matrix=np.array([[.9,.8],[.7,.1]])
    def overlap(box, old):return matrix[int(old[0]),int(round((box[0]-.1)*10))]
    pairs=assign(tracks,detections,.05,.3,AssociationConfig(method='iou_hungarian'),overlap)
    assert set(pairs)=={(0,1),(1,0)}
    greedy=assign(tracks,detections,.05,.3,AssociationConfig(),overlap)
    assert greedy==[(0,0)]


def test_eviction_clears_identity_velocity_and_feature_history():
    b=ObjectBank(1,3,2,overflow='evict_oldest_invisible',association={'method':'appearance_motion'})
    b.update(0,0.,[obs(.1)]);b.update(1,.05,[obs(.2,(-1,0))])
    assert len(b.tracks)==1 and len(b.tracks[0]['events'])==1
    np.testing.assert_array_equal(b.tracks[0]['velocity'],[0,0])
    np.testing.assert_array_equal(b.tracks[0]['events'][0][2],[-1,0])


def test_new_association_obeys_causal_prefix():
    from momart_object_memory.state import ObservationStream, query_schedule
    frames=[obs(.1+i*.002) for i in range(25)]
    def replay():
        stream=ObservationStream(bank(),lambda x:[x])
        for steps in query_schedule(16,8):
            stream.observe([frames[i] for i in steps],steps,np.array(steps)*.05)
        return stream.bank.snapshot('a3')
    first=replay();frames[17:]=[obs(.8,(-1,0))]*8
    for k,v in first.items():np.testing.assert_array_equal(v,replay()[k])


@pytest.mark.parametrize('value', [{'method':'unknown'},{'max_age_seconds':0},
    {'motion_weight':float('nan')},{'appearance_weight':.9,'motion_weight':.2}])
def test_invalid_protocol_rejected(value):
    with pytest.raises(ValueError):AssociationConfig(**value)
