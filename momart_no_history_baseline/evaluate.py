"""Evaluate a MoMaRT No-History checkpoint with the DP-PTP observation window."""

import argparse
import collections
import multiprocessing
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.dataset.dataset_utils import ImageNormalizer, MinMaxNormalizer
from .model import build_agent


TASK_IDS = {
    "unload_dishwasher_to_dresser": 0,
    "table_cleanup_to_sink": 1,
    "table_cleanup_to_dishwasher": 2,
    "table_setup_from_dishwasher": 3,
    "table_setup_from_dresser": 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--env-name", choices=sorted(TASK_IDS), required=True)
    parser.add_argument("--env-type", choices=("expert", "suboptimal"), required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--num-video-episodes", type=int, default=1)
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def load_normalizers(config, train_dir, device):
    with np.load(pathlib.Path(train_dir) / "normalizers.npz") as stats:
        action = MinMaxNormalizer(
            np.stack([stats["action_min"], stats["action_max"]]),
            device=device)
        proprio = MinMaxNormalizer(
            np.stack([stats["proprio_min"], stats["proprio_max"]]),
            device=device)
    image = ImageNormalizer(device=device)
    obs = {key: image for key in config.shape_meta["obs"] if key != "proprio"}
    obs["proprio"] = proprio
    return {"obs": obs, "action": action}


def make_env_args(cli, config):
    return SimpleNamespace(
        dataset_path=cli.dataset_path,
        raw_shape_meta=OmegaConf.to_container(config.shape_meta, resolve=True),
        shape_meta=OmegaConf.to_container(config.shape_meta, resolve=True),
        render_obs_key="rgb",
        obs_steps=int(config.obs_steps),
        action_steps=int(config.action_steps),
        max_episode_steps=1500,
        num_envs=int(cli.num_envs),
        eval_episodes=int(cli.eval_episodes),
        num_video_episodes=int(cli.num_video_episodes),
        save_video=cli.num_video_episodes > 0,
        work_dir=cli.work_dir,
        seed=int(cli.seed),
    )


def normalize_observations(obs, normalizer, device):
    return {
        key: normalizer["obs"][key].normalize(
            torch.tensor(np.asarray(value, dtype=np.float32),
                         device=device, dtype=torch.float32)).contiguous()
        for key, value in obs.items()
    }


def evaluate(cli, config, agent, normalizer, envs, init_functions):
    if cli.eval_episodes % cli.num_envs:
        raise ValueError("eval_episodes must be divisible by num_envs")
    successes, rewards, steps, infer_times = [], [], [], []
    for batch_index in range(cli.eval_episodes // cli.num_envs):
        selected = init_functions[
            batch_index * cli.num_envs:(batch_index + 1) * cli.num_envs]
        envs.call_each("run_dill_function", args_list=[(fn,) for fn in selected])
        obs = envs.reset()
        total_reward = np.zeros(cli.num_envs, dtype=np.float32)
        success = [False] * cli.num_envs
        elapsed_steps = 0
        agent.reset_cache()
        while elapsed_steps < 1500:
            started = time.time()
            condition = normalize_observations(obs, normalizer, cli.device)
            condition["_task_id"] = torch.full(
                (cli.num_envs,), TASK_IDS[cli.env_name],
                device=cli.device, dtype=torch.long)
            with torch.no_grad():
                agent.reset_cache()
                prior = torch.zeros(
                    cli.num_envs, int(config.horizon), int(config.action_dim),
                    device=cli.device)
                normalized_action, _ = agent.sample(
                    prior=prior, n_samples=cli.num_envs,
                    sample_steps=int(config.sample_steps), solver=None,
                    condition_cfg=condition, w_cfg=1.0,
                    temperature=float(config.temperature), use_ema=True)
            action = normalizer["action"].unnormalize(normalized_action)
            start = int(config.obs_steps) - 1
            action = action[:, start:start + int(config.action_steps)]
            infer_times.append(time.time() - started)
            obs, reward, _, info = envs.step(action.detach().cpu().numpy())
            total_reward += np.asarray(reward, dtype=np.float32)
            for env_index in range(cli.num_envs):
                success[env_index] = success[env_index] or any(
                    item["task"] for item in info[env_index]["sc"])
            elapsed_steps += int(config.action_steps)
            if elapsed_steps % 200 == 0:
                print(f"batch={batch_index + 1} steps={elapsed_steps}/1500 "
                      f"successes={sum(success)}/{cli.num_envs}", flush=True)
        if cli.num_video_episodes:
            envs.render()
        successes.extend(success)
        rewards.extend(total_reward.tolist())
        steps.extend([elapsed_steps] * cli.num_envs)

    step = int(pathlib.Path(cli.checkpoint).stem.split("_")[-1])
    return {
        "method": "no_history", "checkpoint_step": step,
        "env_name": cli.env_name, "env_type": cli.env_type,
        "eval_seed": cli.seed,
        "episode_id": np.arange(len(successes), dtype=np.int64),
        "episode_success": np.asarray(successes, dtype=np.int8),
        "episode_reward": np.asarray(rewards, dtype=np.float32),
        "episode_steps": np.asarray(steps, dtype=np.int32),
        "mean_success": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_infer_time": float(np.mean(infer_times)),
    }


def main():
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    cli = parse_args()
    if cli.disable_cudnn:
        torch.backends.cudnn.enabled = False
    pathlib.Path(cli.work_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    config = OmegaConf.load(cli.train_config)
    mapping = collections.defaultdict(list)
    for key, attr in config.shape_meta["obs"].items():
        mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(mapping)
    normalizer = load_normalizers(
        config, pathlib.Path(cli.train_config).parent.parent, cli.device)
    config.device = cli.device
    config.inference_action_steps = int(config.action_steps)
    agent = build_agent(config, normalizer)
    state = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    agent.model.load_state_dict(state["model"], strict=True)
    agent.model_ema.load_state_dict(state["model_ema"], strict=True)
    del state
    agent.model.eval()
    agent.model_ema.eval()
    from .envs import make_async_envs
    envs, init_functions = make_async_envs(make_env_args(cli, config))
    result = evaluate(cli, config, agent, normalizer, envs, init_functions)
    result_path = pathlib.Path(cli.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(result_path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
