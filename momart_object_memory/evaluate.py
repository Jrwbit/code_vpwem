"""Evaluate any arm through one baseline rollout/score implementation."""
import collections
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
import robomimic.utils.obs_utils as ObsUtils
from momart_boxed_no_history import evaluate as baseline
from .model import build_agent
from .online import OnlineObjects
from .perception import FrozenPerception
from .train import comparison_contract
from .main_visual import main_visual_contract, build_main_boxer


def normalize_observations(obs, normalizer, device):
    objects = {k: torch.as_tensor(np.ascontiguousarray(v), device=device)
               for k, v in obs.items() if k.startswith('_object_')}
    scene = baseline.normalize_observations({k: v for k, v in obs.items()
                                            if not k.startswith('_object_')}, normalizer, device)
    scene.update(objects)
    return scene


def main():
    multiprocessing.set_start_method('spawn', force=True)
    cli = baseline.parse_args()
    cfg = OmegaConf.load(cli.train_config)
    protocol = json.loads((Path(cli.train_config).parent/'object_protocol.json').read_text())
    if (protocol.get('protocol_version') != 'independent_main_visual_v2'
            or protocol.get('main_visual') != main_visual_contract(cfg)
            or comparison_contract(cfg) != protocol['comparison_contract']
            or cfg.object_memory.mode != protocol['mode']):
        raise ValueError('Saved object experiment config differs from protocol')
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    if cli.disable_cudnn:
        torch.backends.cudnn.enabled = False
    mapping = collections.defaultdict(list)
    for key, value in cfg.shape_meta.obs.items():
        mapping[value.get('type','low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(mapping)
    Path(cli.work_dir).mkdir(parents=True, exist_ok=True)
    perceive = FrozenPerception(cfg.object_memory, cli.device)
    if perceive.metadata != protocol['perception']:
        raise ValueError('Online perception differs from training cache')
    # Set runtime device only after validating the saved training contract.
    cfg.device = cli.device; cfg.inference_action_steps = int(cfg.action_steps)
    norm = baseline.load_normalizers(cfg, Path(cli.train_config).parent, cli.device)
    agent = build_agent(cfg, norm)
    state = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    agent.model.load_state_dict(state['model'], strict=True)
    agent.model_ema.load_state_dict(state['model_ema'], strict=True)
    agent.model.eval(); agent.model_ema.eval()
    main_boxer = build_main_boxer(cfg, cli.device, cli.num_envs, cli.precision_bundle)
    cli.rgb_boxer = OnlineObjects(cfg, perceive, cli.num_envs, main_boxer)
    from momart_no_history_baseline.envs import make_async_envs
    envs, inits = make_async_envs(baseline.make_env_args(cli,cfg))
    try:
        result = baseline.evaluate(cli,cfg,agent,norm,envs,inits,
                                   observation_normalizer=normalize_observations)
    finally:
        envs.close()
    result.update(method='object_memory_'+cfg.object_memory.mode,
                  comparison_sha256=protocol['comparison_sha256'], perception=perceive.metadata,
                  main_visual=protocol['main_visual'])
    path = Path(cli.result_path); path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,result)
    print(result)


if __name__ == '__main__':
    main()
