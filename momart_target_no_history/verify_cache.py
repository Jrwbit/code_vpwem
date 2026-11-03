"""Full ten-source integrity plus actual multimodal Dataset consumption checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import zarr
from omegaconf import OmegaConf
from momart_boxed_no_history.train import initialize_modalities
from momart_no_history_baseline.dataset import ZarrNoHistorySequenceDataset
from .dataset import TargetNoHistoryDataset


def main(a):
    cfg=OmegaConf.load(a.config);initialize_modalities(cfg)
    metadata=json.loads((a.cache_root/'target_metadata.json').read_text());records=[]
    for spec in cfg.task_datasets:
        if a.sources and str(spec.name) not in a.sources:continue
        name=str(spec.name);source=Path(cfg.data_root)/(name+'.zarr');path=a.cache_root/(name+'.bbox.zarr')
        root=zarr.open_group(str(path),mode='r');kwargs=dict(zarr_path=source,obs_keys=cfg.shape_meta.obs,seq_length=int(cfg.horizon),obs_steps=int(cfg.obs_steps),subsample_ratio=int(cfg.dataset_memory_subsample_ratio))
        base=ZarrNoHistorySequenceDataset(**kwargs)
        if root.attrs['complete'] is not True or root.attrs['completed_frames']!=len(base):raise ValueError('Incomplete cache: '+name)
        for key,expected in root.attrs['array_sha256'].items():
            if hashlib.sha256(np.asarray(root[key]).tobytes()).hexdigest()!=expected:raise ValueError('Array checksum failed: '+name+'/'+key)
        selected=np.asarray(root['valid']);classes=np.asarray(root['class_id'])
        if (classes[selected]<0).any() or (classes[selected]>5).any() or (classes[~selected]!=-1).any():raise ValueError('Invalid selected class/valid mapping')
        ids=[0,int(base.replay_buffer.episode_ends[0])-1,len(base)//2,len(base)-1]
        for mode in ['box','box_coordinates_roi']:
            data=TargetNoHistoryDataset(**kwargs,bbox_path=path,mode=mode,target_metadata=metadata)
            samples=data.__getitems__(ids)
            for index,sample in zip(ids,samples):
                raw=base[index]
                for key in cfg.shape_meta.obs:
                    if key!='rgb':np.testing.assert_array_equal(sample['obs'][key],raw['obs'][key])
                np.testing.assert_array_equal(sample['actions'],raw['actions'])
                start,end,_,_=data.sampler.indices[index];frames=np.minimum(np.arange(start,start+cfg.obs_steps),end-1)
                for t,frame in enumerate(frames):
                    if not data.valid[frame]:np.testing.assert_array_equal(sample['obs']['rgb'][t],raw['obs']['rgb'][t])
                if mode=='box_coordinates_roi':
                    last=frames[-1]
                    np.testing.assert_array_equal(sample['obs']['_bbox'],data.boxes[last] if data.valid[last] else np.zeros(4))
                    assert bool(sample['obs']['_bbox_valid'])==bool(data.valid[last])
        records.append(dict(source=name,frames=len(base),valid_frames=int(selected.sum()),
            selected_classes={str(c):int((classes==c).sum()) for c in range(6)},both_policy_input_modes_passed=True,non_main_observations_and_actions_unchanged=True,all_array_checksums_passed=True))
    expected=set(a.sources) if a.sources else set(metadata['supported_sources'])
    if {r['source'] for r in records}!=expected or (not a.sources and len(records)!=10):raise ValueError('Requested source coverage differs')
    result=dict(complete=len(records)==10,sources=records,total_frames=sum(r['frames'] for r in records),
        valid_frames=sum(r['valid_frames'] for r in records),bundle_manifest_sha256=metadata['bundle_manifest_sha256'],
        caveat='Validity is selected-box coverage, not annotation accuracy or policy success; dataset interface verification samples 4 indices/source/mode')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','cache-root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sources',nargs='+')
    main(p.parse_args())
