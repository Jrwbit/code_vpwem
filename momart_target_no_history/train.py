"""Train either historical policy structure using the same new target cache."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from momart_boxed_no_history import train as base
from momart_no_history_baseline.model import build_agent as build_box_agent
from .legacy_tokens import build_agent as build_token_agent
from .dataset import TargetNoHistoryDataset


def build_suite(args):
    metadata = OmegaConf.to_container(args.target_metadata, resolve=True)
    unsupported = {str(s.name) for s in args.task_datasets} - set(metadata['supported_sources'])
    if unsupported:
        raise ValueError('Frozen selector does not cover requested sources: ' + ', '.join(sorted(unsupported)))
    def dataset_type(**kwargs):
        return TargetNoHistoryDataset(**kwargs, mode=args.target_mode, target_metadata=metadata)
    return base.build_suite(args, dataset_type=dataset_type)


def normalize_observations(obs, normalizer, device):
    return {key: (torch.as_tensor(value,device=device,dtype=torch.float32)
                 if key.startswith('_bbox') else
                 torch.as_tensor(value,device=device,dtype=torch.float32)*2-1
                 if key == '_roi' else normalizer['obs'][key].normalize(
                     torch.as_tensor(value,device=device,dtype=torch.float32)))
            for key,value in obs.items()}


def split_batch(args, dataset, task_id, batch):
    condition = normalize_observations(batch['obs'], dataset.normalizer, args.device)
    condition['_task_id'] = torch.full((len(batch['actions']),),int(task_id),device=args.device,dtype=torch.long)
    return condition, dataset.normalizer['action'].normalize(batch['actions'].to(args.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    cli = parser.parse_args()
    args = OmegaConf.load(cli.config)
    if args.horizon != args.obs_steps + args.action_steps - 1:
        raise ValueError('Inconsistent observation/action horizon')
    if args.target_mode not in ('box','box_coordinates_roi'):
        raise ValueError('Unknown target input mode')
    builder = build_box_agent if args.target_mode == 'box' else build_token_agent
    base.train(args,suite_builder=build_suite,agent_builder=builder,batch_splitter=split_batch)


if __name__ == '__main__':
    main()
