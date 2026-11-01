"""Partition independent datasets across GPUs; build shared caches without training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def partition_datasets(frame_counts, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError('Distinct nonnegative GPU IDs required')
    queues = {gpu: [] for gpu in gpus}
    loads = dict.fromkeys(gpus, 0)
    for name, frames in sorted(frame_counts.items(), key=lambda item: (-item[1], item[0])):
        gpu = min(gpus, key=lambda g: (loads[g], g))
        queues[gpu].append(name)
        loads[gpu] += frames
    return queues, loads


def save_status(output, **values):
    values.update(updated_unix=time.time(), supervisor_pid=os.getpid(), policy_training_started=False)
    temp = output/'status.tmp'
    temp.write_text(json.dumps(values, indent=2)+'\n')
    temp.replace(output/'status.json')


def supervise(configs, output):
    children = {}
    logs = []
    try:
        for gpu, path in configs.items():
            log = (output/f'gpu{gpu}.log').open('a')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', WANDB_MODE='offline')
            children[gpu] = subprocess.Popen(
                [sys.executable, '-m', 'momart_object_memory.prepare_train',
                 '--config', str(path), '--device', 'cuda:0', '--prepare-only'],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        pids = {gpu: child.pid for gpu, child in children.items()}
        save_status(output, stage='preparing', pids=pids)
        while True:
            codes = {gpu: child.poll() for gpu, child in children.items()}
            if all(code is not None for code in codes.values()):
                stage = 'cache_complete' if all(code == 0 for code in codes.values()) else 'failed'
                save_status(output, stage=stage, pids=pids, exit_codes=codes)
                if stage == 'failed':
                    raise RuntimeError('Cache workers failed; successful independent shards retained: '+str(codes))
                return
            time.sleep(5)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()


def main():
    from omegaconf import OmegaConf
    import zarr
    from momart_no_history_baseline.cache import training_replay
    from .cache import replay_signature
    from .perception import file_sha256, mask_perception_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', default='momart_object_memory/config.yaml')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if not (bundle/'READY').is_file():
        raise ValueError('Bundle is not ready')
    manifest = json.loads((bundle/'bundle_manifest.json').read_text())
    source = Path(__file__).resolve().parents[1]
    for name, digest in manifest['source_sha256'].items():
        if file_sha256(source/name) != digest:
            raise ValueError('Production source differs from bundle: '+name)
    if file_sha256(bundle/'input_manifest.json') != manifest['files']['input_manifest.json']:
        raise ValueError('Input manifest checksum differs')
    inputs = json.loads((bundle/'input_manifest.json').read_text())['datasets']
    cfg = OmegaConf.load(args.base_config)
    cfg.data_root = str(args.data_root.resolve())
    cfg.object_memory.cache_root = str(args.cache_root.resolve())
    cfg.object_memory.detector_checkpoint = str(bundle/'assets/mask.pt')
    cfg.object_memory.feature_checkpoint = str(bundle/'assets/roi_source.pt')
    mask_perception_contract(cfg.object_memory)
    counts = {}
    for spec in cfg.task_datasets:
        name = str(spec.name)
        replay, names, mapping = training_replay(zarr.open_group(str(Path(cfg.data_root)/(name+'.zarr')), mode='r'))
        actual = replay_signature(replay, mapping, names)
        if actual != inputs[name]['replay']:
            raise ValueError('Original BHR replay identity differs: '+name)
        counts[name] = actual['frames']
    queues, loads = partition_datasets(counts, args.gpus)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = {}
    for gpu, names in queues.items():
        if not names:
            continue
        arm = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        arm.task_datasets = [OmegaConf.to_container(spec, resolve=True) for spec in cfg.task_datasets if spec.name in names]
        # Preserve the longest-first order used in the partition plan.
        arm.task_datasets = sorted(arm.task_datasets, key=lambda spec: names.index(str(spec.name)))
        arm.work_dir = str(output/f'gpu{gpu}')
        Path(arm.work_dir).mkdir()
        path = output/f'gpu{gpu}.yaml'
        OmegaConf.save(arm, path)
        configs[gpu] = path
    plan = dict(gpus=args.gpus, queues=queues, frames_per_gpu=loads, total_frames=sum(counts.values()),
                cache_root=cfg.object_memory.cache_root, data_root=cfg.data_root,
                python=sys.executable, bundle_commit=manifest['code_commit'],
                raw_arrays='Original BHR data; replay identity checked, no relocation',
                policy_training_started=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    supervise(configs, output)


if __name__ == '__main__':
    main()
