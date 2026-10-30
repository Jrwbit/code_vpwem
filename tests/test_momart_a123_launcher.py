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
