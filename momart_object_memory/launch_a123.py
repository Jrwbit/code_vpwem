"""Prepare one verified Mask cache and supervise three parallel BS256 runs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from omegaconf import OmegaConf
import zarr
from .cache import replay_signature
from .perception import file_sha256, mask_perception_contract
from .train import comparison_contract
from momart_no_history_baseline.cache import training_replay


def save_status(path, **values):
    values.update(updated_unix=time.time())
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(values,indent=2)+'\n');temp.replace(path)


def array_hash(array):
    digest=hashlib.sha256()
    size=max(1,16*2**20//max(1,int(np.prod(array.shape[1:]))*array.dtype.itemsize))
    for start in range(0,len(array),size):
        digest.update(np.ascontiguousarray(array[start:start+size]).tobytes())
    return digest.hexdigest()


def verify_and_relocate(config, bundle, bbox_root):
    """Rebind only copied bowl metadata, after full raw-array verification.

    Neither source data nor the original bowl caches are edited. This verifies
    lossless bytes, episode ordering and frame mapping before moving paths.
    """
    manifest=json.loads((bundle/'input_manifest.json').read_text())
    for spec in config.task_datasets:
        name=str(spec.name);source=Path(config.data_root)/(name+'.zarr')
        root=zarr.open_group(str(source),mode='r')
        replay,names,mapping=training_replay(root)
        expected=manifest['datasets'][name]
        if replay_signature(replay,mapping,names)!=expected['replay']:
            raise ValueError('Training episode/frame mapping differs: '+name)
        for key,value in expected['arrays'].items():
            array=root[key]
            if (list(array.shape)!=value['shape'] or array.dtype.str!=value['dtype']
                    or array_hash(array)!=value['sha256']):
                raise ValueError('Training raw array differs: '+name+'/'+key)
        original=bundle/'bbox'/(name+'.bbox.zarr');target=bbox_root/(name+'.bbox.zarr')
        shutil.copytree(original,target)
        boxes=zarr.open_group(str(target),mode='a');signature=dict(boxes.attrs['signature'])
        if (signature['shape']!=list(replay['rgb'].shape)
                or signature['episode_ends_sha256']!=expected['replay']['episode_ends_sha256']
                or file_sha256(target/'.zattrs')!=expected['bbox_zattrs_sha256']):
            raise ValueError('Bowl cache differs before relocation: '+name)
        for key in ('boxes','valid'):
            if array_hash(boxes[key])!=expected['bbox_arrays'][key]:
                raise ValueError('Bowl cache array differs: '+name+'/'+key)
        signature['source']=str(source.resolve());boxes.attrs['signature']=signature


def make_configs(base, output, bundle, data_root, cache_root, steps=250000):
    if steps<1:raise ValueError('Positive gradient steps required')
    cfg=copy.deepcopy(base)
    cfg.data_root=str(data_root.resolve());cfg.bbox_root=str(output/'bbox')
    cfg.object_memory.cache_root=str(cache_root.resolve())
    cfg.object_memory.detector_checkpoint=str(bundle/'assets/mask.pt')
    cfg.object_memory.feature_checkpoint=str(bundle/'assets/roi_source.pt')
    cfg.main_visual.precision_bundle=str(bundle/'assets/main_bundle.json')
    cfg.device='cuda:0';cfg.seed=100;cfg.gradient_steps=steps
    cfg.physical_batch_size=256;cfg.effective_batch_size=256;cfg.num_workers=1
    cfg.log_freq=100;cfg.save_freq=min(10000,steps)
    cfg.keep_checkpoints=max(10,steps//cfg.save_freq)
    configs=[]
    for mode in ('a1','a2','a3'):
        arm=copy.deepcopy(cfg);arm.object_memory.mode=mode
        arm.work_dir=str(output/'runs'/mode/'seed100');configs.append(arm)
    if any(comparison_contract(c)!=comparison_contract(configs[0]) for c in configs):
        raise ValueError('Uncontrolled arm config difference')
    return configs


def child_environment(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',PYTHONUNBUFFERED='1')
    return env


def supervise(config_paths, gpus, output):
    """Launch all arms before waiting; compare initialization before continuing."""
    children=[];logs=[];verified=False
    try:
        for path,gpu in zip(config_paths,gpus):
            cfg=OmegaConf.load(path);log=(output/(cfg.object_memory.mode+'.log')).open('a')
            logs.append(log)
            command=[sys.executable,'-m','momart_object_memory.train','--config',str(path),
                     '--mode',cfg.object_memory.mode,'--device','cuda:0']
            child=subprocess.Popen(command,env=child_environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
        save_status(output/'status.json',stage='training_starting',pids=[c.pid for c in children],gpus=gpus)
        while True:
            codes=[c.poll() for c in children]
            if any(code is not None and code!=0 for code in codes):
                raise RuntimeError('An arm failed; inspect logs: '+str(codes))
            runs=[Path(OmegaConf.load(p).work_dir) for p in config_paths]
            if not verified and all((r/'initialization.json').exists() for r in runs):
                with (output/'comparison.json').open('w') as target:
                    subprocess.run([sys.executable,'-m','momart_object_memory.compare',*map(str,runs)],
                                   stdout=target,check=True)
                verified=True
                save_status(output/'status.json',stage='training',pids=[c.pid for c in children],
                            gpus=gpus,common_initialization_verified=True,steps=OmegaConf.load(config_paths[0]).gradient_steps)
            if all(code is not None for code in codes):
                if not verified:raise RuntimeError('Training exited before initialization comparison')
                save_status(output/'status.json',stage='complete',exit_codes=codes)
                return
            time.sleep(5)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',default='momart_object_memory/config.yaml')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs=3,type=int,required=True,help='One device per arm; repeat IDs only after capacity checks')
    p.add_argument('--steps',type=int,default=250000)
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        manifest=json.loads((args.bundle/'bundle_manifest.json').read_text())
        for name,digest in manifest['files'].items():
            path=(args.bundle/name).resolve();path.relative_to(args.bundle.resolve())
            if file_sha256(path)!=digest:raise ValueError('Bundle checksum differs: '+name)
        source_root=Path(__file__).resolve().parents[1]
        for name,digest in manifest['source_sha256'].items():
            if file_sha256(source_root/name)!=digest:
                raise ValueError('Launched source differs from Git bundle: '+name)
        configs=make_configs(OmegaConf.load(args.base_config),output,args.bundle.resolve(),
                             args.data_root,args.cache_root,args.steps)
        mask_perception_contract(configs[0].object_memory)
        if configs[0].main_visual.precision_bundle_sha256!=file_sha256(configs[0].main_visual.precision_bundle):
            raise ValueError('Archived main visual bundle differs')
        save_status(output/'status.json',stage='verifying_lossless_inputs',scope='engineering_comparison_not_gate_b_admission')
        verify_and_relocate(configs[0],args.bundle.resolve(),Path(configs[0].bbox_root))
        paths=[]
        for cfg in configs:
            path=output/(cfg.object_memory.mode+'.yaml');OmegaConf.save(cfg,path);paths.append(path)
        save_status(output/'status.json',stage='preparing_shared_mask_cache')
        with (output/'prepare.log').open('w') as log:
            subprocess.run([sys.executable,'-m','momart_object_memory.prepare_train','--config',str(paths[0]),
                            '--device','cuda:0','--prepare-only'],env=child_environment(args.gpus[0]),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        supervise(paths,args.gpus,output)
    except BaseException as error:
        save_status(output/'status.json',stage='failed',error=str(error))
        raise


if __name__=='__main__':main()
