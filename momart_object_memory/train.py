"""A1/A2/A3 share the archived baseline optimizer loop and data normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from momart_boxed_no_history import train as baseline
from .dataset import ObjectDataset
from .model import build_agent
from .perception import file_sha256


def comparison_contract(args):
    value = OmegaConf.to_container(args, resolve=True)
    value.pop('work_dir', None)
    value['object_memory'].pop('mode')
    # Each training seed has its own comparison group; no other differences allowed.
    return value


def build_suite(args):
    datasets, tasks, norm, _, stats = baseline.build_suite(args)
    wrapped = {name: ObjectDataset(ds, args, name) for name, ds in datasets.items()}
    metadata = [ds.perception_metadata for ds in wrapped.values()]
    if any(m != metadata[0] for m in metadata):
        raise ValueError('Dataset perception versions differ')
    if metadata[0]['feature_checkpoint_sha256'] != file_sha256(args.object_memory.feature_checkpoint):
        raise ValueError('Feature checkpoint differs from preprocessing cache')
    if metadata[0]['threshold'] != float(args.object_memory.threshold):
        raise ValueError('Detector threshold differs from cache')
    loaders = {}
    for name, ds in wrapped.items():
        sampler = ds.get_dataset_sampler()
        loaders[name] = torch.utils.data.DataLoader(
            ds, sampler=sampler, shuffle=sampler is None, batch_size=int(args.physical_batch_size),
            drop_last=True, num_workers=int(args.num_workers), pin_memory=True,
            persistent_workers=int(args.num_workers)>0)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    contract = comparison_contract(args)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    source_root = Path(__file__).resolve().parent.parent
    source_paths = sorted(Path(__file__).parent.glob('*.py')) + [
        source_root/'momart_boxed_no_history/train.py',
        source_root/'momart_boxed_no_history/evaluate.py',
        source_root/'momart_no_history_baseline/model.py',
        source_root/'cleandiffuser/diffusion/longmem_ddpm.py',
        source_root/'cleandiffuser/nn_diffusion/longmem_chitransformerptp.py',
        source_root/'cleandiffuser/nn_condition/multi_image_condition.py']
    source_hashes = {str(p.relative_to(source_root)):file_sha256(p) for p in source_paths}
    manifest = dict(mode=args.object_memory.mode, comparison_sha256=fingerprint,
                    source_sha256=source_hashes,
                    comparison_contract=contract, perception=metadata[0])
    path = work/'object_protocol.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to mix experiment configurations in one output directory')
    if list((work/'models').glob('model_*.pt')):
        raise ValueError('Automatic resume disabled: full RNG/loader resume needs separate validation')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return wrapped, tasks, norm, loaders, stats


def split_batch(args, dataset, task_id, batch):
    condition, action = baseline.split_batch(args, dataset.base, task_id, batch)
    condition.update({'_object_'+k: v.to(args.device, non_blocking=True)
                      for k, v in batch['objects'].items()})
    return condition, action


def build_training_agent(args, normalizer):
    return build_agent(args, normalizer, record_initialization=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--mode', choices=('a1','a2','a3','a3_latest_control'), required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.object_memory.mode = args.mode
    if args.device:
        cfg.device = args.device
    if cfg.horizon < cfg.obs_steps + cfg.action_steps - 1:
        raise ValueError('Action execution exceeds predicted horizon')
    baseline.train(cfg, suite_builder=build_suite, agent_builder=build_training_agent, batch_splitter=split_batch)


if __name__ == '__main__':
    main()
