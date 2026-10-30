"""Build a BHR-to-SH runtime bundle; datasets/weights never enter Git."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
import zarr
from momart_object_memory.launch_a123 import array_hash
from momart_object_memory.cache import replay_signature
from momart_object_memory.perception import file_sha256, mask_perception_contract
from momart_no_history_baseline.cache import training_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='momart_object_memory/config.yaml')
    p.add_argument('--code-archive',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.load(args.config);mask_perception_contract(cfg.object_memory)
    (out/'assets').mkdir()
    for source,target in [(cfg.object_memory.detector_checkpoint,'mask.pt'),
                          (cfg.object_memory.feature_checkpoint,'roi_source.pt'),
                          (cfg.main_visual.precision_bundle,'main_bundle.json')]:
        shutil.copyfile(source,out/'assets'/target)
    shutil.copyfile(args.code_archive,out/'code.tar')
    datasets={}
    for spec in cfg.task_datasets:
        name=str(spec.name);root=zarr.open_group(str(Path(cfg.data_root)/(name+'.zarr')),mode='r')
        replay,names,mapping=training_replay(root)
        target=out/'bbox'/(name+'.bbox.zarr')
        shutil.copytree(Path(cfg.bbox_root)/(name+'.bbox.zarr'),target)
        boxes=zarr.open_group(str(target),mode='r')
        arrays={}
        for group in ('data','meta'):
            for key,array in root[group].items():
                arrays[group+'/'+key]=dict(shape=list(array.shape),dtype=array.dtype.str,sha256=array_hash(array))
        datasets[name]=dict(replay=replay_signature(replay,mapping,names),arrays=arrays,
            bbox_zattrs_sha256=file_sha256(target/'.zattrs'),
            bbox_arrays={key:array_hash(boxes[key]) for key in ('boxes','valid')})
        print(json.dumps(dict(dataset=name,raw_arrays_hashed=len(arrays))),flush=True)
    (out/'input_manifest.json').write_text(json.dumps(dict(datasets=datasets),indent=2)+'\n')
    root=Path(__file__).resolve().parents[1]
    sources={str(path.relative_to(root)):file_sha256(path) for path in (root/'momart_object_memory').glob('*.py')}
    files={str(path.relative_to(out)):file_sha256(path) for path in out.rglob('*') if path.is_file()}
    (out/'bundle_manifest.json').write_text(json.dumps(dict(code_commit=args.commit,source_sha256=sources,
        files=files,steps=250000,physical_batch_size=256,scope='engineering_comparison_not_gate_b_admission'),indent=2)+'\n')
    (out/'READY').write_text('Complete runtime bundle; verify SH data and environment before launch.\n')
    print(json.dumps(dict(ready=True,bundle=str(out),code_commit=args.commit)),flush=True)


if __name__=='__main__':main()
