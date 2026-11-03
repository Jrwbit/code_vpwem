"""Produce two explicit replacement configs; never drop old tasks/modalities."""
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf


def replacement_configs(base, metadata, cache_root, output_root):
    if metadata.get('annotation_standard') != 'causal_selected_target_v1':
        raise ValueError('Expected frozen target metadata')
    configs = {}
    for mode in ('box', 'box_coordinates_roi'):
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.target_mode = mode
        cfg.target_metadata = metadata
        cfg.bbox_root = str(cache_root)
        cfg.history_method = 'selected_target_' + mode
        cfg.work_dir = str(Path(output_root) / mode / 'seed${seed}')
        configs[mode] = cfg
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config',type=Path,required=True)
    parser.add_argument('--target-metadata',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,required=True)
    cli = parser.parse_args()
    metadata = json.loads(cli.target_metadata.read_text())
    configs = replacement_configs(OmegaConf.load(cli.base_config),metadata,cli.cache_root,cli.output_root)
    cli.config_dir.mkdir(parents=True,exist_ok=False)
    for mode,cfg in configs.items():
        OmegaConf.save(cfg,cli.config_dir/(mode+'.yaml'))
    missing = sorted({str(s.name) for s in configs['box'].task_datasets} - set(metadata['supported_sources']))
    print(json.dumps(dict(configs_written=True,unsupported_sources=missing,
        ready_to_train=not missing,requires_complete_matching_target_caches=True),indent=2))


if __name__ == '__main__':
    main()
