from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf
from momart_target_no_history.train_unified import FAMILIES,family,remap
from momart_target_no_history import selector as selector_module
from momart_target_no_history.prepare_cache import public_frame
from runtime import FrameFeatures,SingleBoxSelector


def test_all_ten_sources_use_one_runtime_with_five_distinct_task_conditions(monkeypatch):
    constructed=[];calls=[]
    class Runtime:
        def __init__(self,*args):self.tasks=FAMILIES;constructed.append(self)
        def select(self,frame,ep,step):calls.append((self,frame,ep,step));return {'valid':False}
    monkeypatch.setattr(selector_module,'SingleBoxSelector',Runtime)
    selector=selector_module.UnifiedTargetSelector(Path('/frozen/one/model'))
    frame=FrameFeatures(np.zeros((8,139),np.float32),np.arange(280,dtype=np.float32),np.zeros(8,bool),'ignored')
    for i,task in enumerate(FAMILIES):
        for style in ('expert','suboptimal'):
            source=task+'_'+style;selector.select(frame,source,source+'/demo_1',3)
            actual=calls[-1][1]
            np.testing.assert_array_equal(actual.context[:276],frame.context[:276])
            np.testing.assert_array_equal(actual.context[276:],np.eye(5)[i])
            assert actual.task==task and calls[-1][0] is constructed[0]
    assert len(constructed)==1 and len(calls)==10
    with pytest.raises(ValueError,match='Unsupported'):family('unknown')


def test_joint_training_remaps_quality_suffix_without_changing_sensor_context():
    tasks=[FAMILIES[0]+'_expert',FAMILIES[4]+'_expert',FAMILIES[4]+'_suboptimal']
    context=np.c_[np.arange(3*276).reshape(3,276),np.eye(3)].astype(np.float32)
    mapped=remap(dict(context=context,tasks=tasks))
    np.testing.assert_array_equal(mapped['context'][:,:276],context[:,:276])
    np.testing.assert_array_equal(mapped['context'][:,276:],np.eye(5)[[0,4,4]])
    assert mapped['tasks']==FAMILIES


@pytest.mark.parametrize('tasks',[['a','b','c','d'],FAMILIES])
def test_runtime_preserves_four_task_support_and_accepts_frozen_five_task_schema(tasks):
    net=object.__new__(SingleBoxSelector);net.tasks=tasks;net.component={};net.calibration={'threshold':.6,'margin':0};net.reset()
    net.probability=lambda *args:np.array([.9,0,0,0,0,0,0,0,.1],np.float32)
    c=np.zeros((8,139),np.float32);c[0,128:132]=[.1,.2,.6,.8];c[0,133]=1
    context=np.r_[np.zeros(276),np.eye(len(tasks))[0]].astype(np.float32)
    f=FrameFeatures(c,context,np.array([True]+[False]*7),tasks[0])
    result=net.select(f,'episode',0);assert result['slot']==0
    np.testing.assert_array_equal(net.select(f,'episode',0)['box'],result['box'])
    f.context=np.zeros(len(context)+1,np.float32)
    with pytest.raises(ValueError,match='context'):net.select(f,'new',0)


def test_public_frame_only_uses_explicit_current_and_past_robot_sensors():
    class Geometry:
        def project(self,proprio,torso,fov):return np.array([1.,0.,0.]),(np.array([.5,.5]),True)
    obs=SimpleNamespace(feature=np.zeros(128),box=np.array([.2,.2,.7,.7]),score=.9,class_id=0)
    current=np.arange(10,dtype=np.float32);past=current-1
    frame=public_frame(Geometry(),[obs],np.ones(128),np.full(128,2),current,past,.2,.1,np.ones((90,120,1)),45.,3.)
    np.testing.assert_array_equal(frame.context[:128],1)
    np.testing.assert_array_equal(frame.context[128:256],2)
    np.testing.assert_array_equal(frame.context[256:266],current)
    np.testing.assert_array_equal(frame.context[266:276],current-past)
    assert frame.valid.sum()==1 and frame.geometry_candidates.shape==(8,16)


def test_cache_commits_complete_prefix_resumes_and_keeps_episode_t_minus_8_boundary(tmp_path,monkeypatch):
    import h5py
    from test_momart_batch_sequence_reads import make_dataset
    from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
    from momart_object_memory.prepare_train import encode
    from momart_object_memory.state import ObjectObservation
    from momart_target_no_history import prepare_cache as module
    d=make_dataset(ZarrNoHistorySequenceDataset);d.source_frame_map=np.arange(6);d._demo_id_to_start_indices={'demo_0':0,'demo_1':1}
    source=tmp_path/'test.zarr';hdf_path=tmp_path/'test.hdf5'
    zarr.open_group(str(source),mode='w').attrs['source_path']=str(hdf_path)
    with h5py.File(hdf_path,'w') as f:
        data=f.create_group('data');data.attrs['env_args']=json.dumps({'env_kwargs':{'ig_config':{'vertical_fov':45,'depth_high':3}}})
        for demo,length in [('demo_0',1),('demo_1',5)]:
            ep=data.create_group(demo);obs=ep.create_group('obs');obs['proprio']=np.zeros((length,10),np.float32);obs['depth']=np.ones((length,8,9,1),np.float32);ep['states']=np.zeros((length,14))
    global_values=np.arange(6*128,dtype=np.float32).reshape(6,128)
    np.save(tmp_path/'test.global.npy',global_values)
    meta={'sensor_protocol_sha256':'x'}
    (tmp_path/'test.global.json').write_text(json.dumps({'complete':True,'signature':{'sensor_protocol_sha256':'x'}}))
    (tmp_path/'test.robot_layout.json').write_text(json.dumps({'offset':0}))
    with sqlite3.connect(str(tmp_path/'test.perception.sqlite')) as con:
        con.execute('CREATE TABLE observations(frame INTEGER PRIMARY KEY,payload BLOB)')
        item=ObjectObservation(0,np.array([.1,.2,.7,.8],np.float32),.9,np.zeros(128,np.float32))
        con.executemany('INSERT INTO observations VALUES (?,?)',[(i,encode([item],128)) for i in range(6)])
    class Geometry:
        movable=[0];joints={'torso_lift_joint':0}
        def __init__(self,*args):pass
        def project(self,*args):return np.array([1.,0.,0.]),(np.array([.5,.5]),True)
        def close(self):pass
    calls=[];interrupt={'enabled':True}
    class Selector:
        def __init__(self,*args):self.runtime=SimpleNamespace(component={'kind':'parent','screen':'weights','name':'model'},load=lambda path:{'spec':{'kind':'mlp'}})
        def select(self,frame,source,episode_id,step):
            calls.append((episode_id,step,frame.context[128:256].copy()))
            if interrupt['enabled'] and len(calls)==3:raise RuntimeError('simulated interruption')
            return {'valid':True,'confidence':.9,'box':np.array([.1,.2,.7,.8]),'class_id':0}
    monkeypatch.setattr(module,'RobotGeometry',Geometry);monkeypatch.setattr(module,'UnifiedTargetSelector',Selector)
    target=tmp_path/'test.bbox.zarr'
    def run():return module.write_locked(d,source,target,'test',tmp_path,tmp_path,Path('/bundle'),Path('/robot'),meta)
    with pytest.raises(RuntimeError,match='interruption'):run()
    assert zarr.open_group(str(target),mode='r').attrs['completed_frames']==1
    interrupt['enabled']=False;calls.clear();result=run()
    root=zarr.open_group(str(target),mode='r')
    assert result['frames']==6 and root.attrs['complete'] and root.attrs['completed_frames']==6
    assert root['valid'][:].all() and (root['class_id'][:]==0).all()
    # Past must clamp to this episode's start (global frame1), not global0.
    for ep,step,past in calls:np.testing.assert_array_equal(past,global_values[1])
    assert run()['already_complete']
