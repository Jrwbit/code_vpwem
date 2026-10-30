import copy
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from momart_object_memory.launch_a123 import make_configs, supervise, child_environment
from momart_object_memory.train import comparison_contract


def test_bs256_configs_share_everything_except_mode_and_output(tmp_path):
    base=OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml')
    configs=make_configs(base,tmp_path,tmp_path/'bundle',tmp_path/'data',tmp_path/'cache')
    assert [c.object_memory.mode for c in configs]==['a1','a2','a3']
    assert len({c.work_dir for c in configs})==3
    for cfg in configs:
        assert cfg.physical_batch_size==cfg.effective_batch_size==256
        assert cfg.gradient_steps==250000 and cfg.device=='cuda:0' and cfg.seed==100
        assert comparison_contract(cfg)==comparison_contract(configs[0])
    assert base.physical_batch_size==64
    assert child_environment(2)['CUDA_VISIBLE_DEVICES']=='2'


def test_parallel_launch_failure_terminates_siblings(tmp_path,monkeypatch):
    configs=make_configs(OmegaConf.load(Path(__file__).parents[1]/'momart_object_memory/config.yaml'),
                         tmp_path,tmp_path/'bundle',tmp_path/'data',tmp_path/'cache')
    paths=[]
    for c in configs:
        path=tmp_path/(c.object_memory.mode+'.yaml');OmegaConf.save(c,path);paths.append(path)
    children=[]
    class Child:
        def __init__(self,*args,**kwargs):
            self.pid=len(children)+1;self.code=2 if self.pid==2 else None;children.append(self)
        def poll(self):return self.code
        def terminate(self):self.code=-15
        def wait(self,timeout=None):return self.code
    monkeypatch.setattr('momart_object_memory.launch_a123.subprocess.Popen',Child)
    with pytest.raises(RuntimeError,match='arm failed'):supervise(paths,[0,1,2],tmp_path)
    assert len(children)==3 and [c.code for c in children]==[-15,2,-15]


def relocation_case(tmp_path):
    import json
    import numpy as np
    import zarr
    from momart_object_memory.cache import replay_signature
    from momart_object_memory.launch_a123 import array_hash
    from momart_object_memory.perception import file_sha256
    from momart_no_history_baseline.cache import training_replay
    data=tmp_path/'data';bundle=tmp_path/'bundle';name='sample'
    root=zarr.open_group(str(data/(name+'.zarr')),mode='w')
    root.attrs.update(vpwem_cache_complete=True,vpwem_cache_kind='momart',demo_names=['demo_0'])
    root.create_dataset('meta/episode_ends',data=np.array([3],np.int64))
    root.create_dataset('data/rgb',data=np.zeros((3,90,120,3),np.uint8))
    root.create_dataset('data/action',data=np.zeros((3,10),np.float32))
    replay,names,mapping=training_replay(root);signature=replay_signature(replay,mapping,names)
    box_path=bundle/'bbox'/(name+'.bbox.zarr');boxes=zarr.open_group(str(box_path),mode='w')
    boxes.attrs.update(complete=True,signature=dict(source='/old/source/sample.zarr',
        shape=[3,90,120,3],episode_ends_sha256=signature['episode_ends_sha256']))
    boxes.create_dataset('boxes',data=np.tile([.1,.1,.2,.2],(3,1)).astype(np.float32))
    boxes.create_dataset('valid',data=np.ones(3,bool))
    expected=dict(replay=signature,bbox_zattrs_sha256=file_sha256(box_path/'.zattrs'),
        bbox_arrays={key:array_hash(boxes[key]) for key in ('boxes','valid')},
        arrays={group+'/'+key:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=array_hash(a))
                for group in ('data','meta') for key,a in root[group].items()})
    (bundle/'input_manifest.json').write_text(json.dumps(dict(datasets={name:expected})))
    cfg=OmegaConf.create(dict(data_root=str(data),task_datasets=[dict(name=name)]))
    return cfg,bundle,root,box_path


def test_relocation_verifies_raw_bytes_and_preserves_original_boxes(tmp_path):
    import json
    from momart_object_memory.launch_a123 import verify_and_relocate
    cfg,bundle,root,box_path=relocation_case(tmp_path)
    original=(box_path/'.zattrs').read_bytes()
    target=tmp_path/'relocated'
    verify_and_relocate(cfg,bundle,target)
    assert (box_path/'.zattrs').read_bytes()==original
    relocated=json.loads((target/'sample.bbox.zarr'/'.zattrs').read_text())
    assert relocated['signature']['source']==str((Path(cfg.data_root)/'sample.zarr').resolve())


def test_relocation_rejects_changed_rgb_before_rebinding(tmp_path):
    from momart_object_memory.launch_a123 import verify_and_relocate
    cfg,bundle,root,box_path=relocation_case(tmp_path)
    root['data/rgb'][0,0,0,0]=1
    with pytest.raises(ValueError,match='raw array differs'):
        verify_and_relocate(cfg,bundle,tmp_path/'relocated')
    assert not (tmp_path/'relocated'/'sample.bbox.zarr').exists()
