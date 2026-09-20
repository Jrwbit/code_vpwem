"""Online MoMaRT evaluation for fair-history multi-task embedding policies."""

import argparse
import collections
import copy
import multiprocessing
import os
import pathlib
import re
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.dataset.dataset_utils import ImageNormalizer, MinMaxNormalizer
from cleandiffuser.nn_condition.multi_image_condition import (
    MultiImageObsConditionMomart,
)
from pipelines.fair_history_momart_multitask_emb import build_agent
from pipelines.dpptp_momart_image_eval import (
    make_async_envs as make_dpptp_async_envs,
)
from pipelines.longmemdpptp_momart_image_emb_eval import (
    DictToObj,
    make_async_envs as make_longmem_async_envs,
)
import cleandiffuser.utils.robomimic_train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils


TASK_IDS = {
    "unload_dishwasher_to_dresser": 0,
    "table_cleanup_to_sink": 1,
    "table_cleanup_to_dishwasher": 2,
    "table_setup_from_dishwasher": 3,
    "table_setup_from_dresser": 4,
}

RAW_SHAPE_META = {
    "action": {"shape": [10]},
    "obs": {
        "rgb": {"shape": [3, 90, 120], "type": "rgb"},
        "rgb_wrist": {"shape": [3, 90, 120], "type": "rgb"},
        "depth": {"shape": [1, 90, 120], "type": "depth"},
        "depth_wrist": {"shape": [1, 90, 120], "type": "depth"},
        "proprio": {"shape": [10], "type": "low_dim"},
        "scan": {"shape": [1, 100], "type": "scan"},
    },
}

# Exact extrema used by fair_history_momart_multitask_emb.py over all ten
# frozen-embedding training datasets (727,192 actions).
GLOBAL_ACTION_MIN = np.asarray(
    [-0.75, -0.9483328888888889, -1.0, -1.0, -1.0,
     -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
GLOBAL_ACTION_MAX = np.asarray(
    [0.75, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
GLOBAL_PROPRIO_STATS = np.asarray([
    [-0.021992577239871025, -0.2502044439315796, -1.0,
     0.1976921558380127, -0.3571072816848755, 0.48243996500968933,
     -0.8407337665557861, -0.8945220708847046, -0.7730650305747986,
     1.4294874972620164e-06],
    [0.006118187680840492, 1.1451035737991333, 1.0,
     0.7982087135314941, 0.3592052459716797, 1.3833026885986328,
     0.9001737833023071, 0.9639379382133484, 0.73549884557724,
     0.9380683898925781],
], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--context-checkpoint")
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
    parser.add_argument(
        "--disable-cudnn", action="store_true",
        help="Use native CUDA convolution kernels when cuDNN cannot share a GPU.")
    return parser.parse_args()


def checkpoint_step(cli):
    if cli.checkpoint_step is not None:
        return int(cli.checkpoint_step)
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(cli.checkpoint))
    if match is None:
        raise ValueError(
            "cannot infer checkpoint step from --checkpoint; pass "
            "--checkpoint-step explicitly")
    return int(match.group(1))


def raw_dataset_config(path, horizon):
    return DictToObj({
        "train": {
            "data": path,
            "num_data_workers": 0,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train",
            "hdf5_validation_filter_key": "valid",
            "seq_length": horizon,
            "pad_seq_length": True,
            "frame_stack": 1,
            "pad_frame_stack": True,
            "dataset_keys": ["actions", "rewards", "dones"],
            "goal_mode": None,
            "cuda": True,
            "batch_size": 16,
        },
        "experiment": {"validate": False},
        "memory": {
            "dataset_memory_type": "random_from_full",
            "dataset_memory_subsample_ratio": 20,
        },
    })


def load_raw_dataset(path, horizon, device, proprio_stats=None):
    dataset, _ = TrainUtils.load_data_for_training(
        raw_dataset_config(path, horizon), obs_keys=RAW_SHAPE_META["obs"])
    if proprio_stats is None:
        proprio_min = torch.full((10,), float("inf"), device=device)
        proprio_max = torch.full((10,), float("-inf"), device=device)
        for demo in dataset.hdf5_cache.values():
            value = torch.as_tensor(
                demo["obs"]["proprio"], device=device, dtype=torch.float32)
            proprio_min = torch.minimum(proprio_min, value.amin(dim=0))
            proprio_max = torch.maximum(proprio_max, value.amax(dim=0))
    else:
        proprio_min = torch.as_tensor(
            proprio_stats[0], device=device, dtype=torch.float32)
        proprio_max = torch.as_tensor(
            proprio_stats[1], device=device, dtype=torch.float32)
    normalizer = collections.defaultdict(dict)
    image = ImageNormalizer(device=device)
    normalizer["obs"].update({
        "rgb": image,
        "rgb_wrist": image,
        "depth": ImageNormalizer(device=device),
        "depth_wrist": ImageNormalizer(device=device),
        "proprio": MinMaxNormalizer(
            torch.stack([proprio_min, proprio_max]), device=device),
        "scan": ImageNormalizer(device=device),
    })
    dataset.normalizer = normalizer
    return dataset


def build_source_encoder(config, checkpoint, device):
    encoder = MultiImageObsConditionMomart(
        shape_meta=RAW_SHAPE_META,
        emb_dim=int(config.embedding_dim),
        rgb_model_name="resnet18",
        resize_shape=None,
        crop_shape=[75, 100],
        random_crop=True,
        use_group_norm=True,
        use_seq=True,
        keep_horizon_dims=True,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    condition = {
        key.removeprefix("condition."): value
        for key, value in state["model_ema"].items()
        if key.startswith("condition.")
    }
    encoder.load_state_dict(condition, strict=True)
    encoder.eval()
    return encoder


def build_policy(config, checkpoint, device):
    config.device = device
    config.inference_action_steps = int(config.action_steps)
    action_stats = np.stack([GLOBAL_ACTION_MIN, GLOBAL_ACTION_MAX])
    action_normalizer = MinMaxNormalizer(action_stats, device=device)
    if "embedding_dim" in config:
        agent = build_agent(config, action_normalizer)
    else:
        from pipelines.fair_history_momart_multitask import (
            build_agent as build_raw_agent,
        )
        normalizer = collections.defaultdict(dict)
        normalizer["action"] = action_normalizer
        agent = build_raw_agent(config, normalizer)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.model.load_state_dict(state["model"], strict=True)
    agent.model_ema.load_state_dict(state["model_ema"], strict=True)
    agent.model.eval()
    agent.model_ema.eval()
    return agent, action_normalizer


def env_args(cli, config):
    args = argparse.Namespace()
    args.dataset_path = os.path.expanduser(cli.dataset_path)
    args.raw_shape_meta = copy.deepcopy(RAW_SHAPE_META)
    args.shape_meta = args.raw_shape_meta
    args.render_obs_key = "rgb"
    args.obs_steps = int(config.obs_steps)
    args.action_steps = int(config.action_steps)
    args.max_episode_steps = 1500
    args.num_envs = int(cli.num_envs)
    args.eval_episodes = int(cli.eval_episodes)
    args.num_video_episodes = int(cli.num_video_episodes)
    args.save_video = cli.num_video_episodes > 0
    args.work_dir = os.path.expanduser(cli.work_dir)
    args.seed = int(cli.seed)
    return args


def normalize_observations(obs, dataset, device):
    normalized = {}
    for key, value in obs.items():
        # Match the official DP-PTP evaluator: materialize each vector-env
        # observation as a fresh contiguous float32 tensor.  The wrapper can
        # return strided NumPy views, which cuDNN's first ResNet convolution
        # does not handle reliably when another process shares the GPU.
        tensor = torch.tensor(
            np.asarray(value, dtype=np.float32),
            device=device,
            dtype=torch.float32,
        )
        normalized[key] = (
            dataset.normalizer["obs"][key].normalize(tensor).contiguous())
    return normalized


def env_factory(history_method):
    """Use the same observation window as the corresponding train system."""
    if str(history_method) == "no_history":
        return make_dpptp_async_envs
    return make_longmem_async_envs


def evaluate(cli, config, dataset, source_encoder, agent, action_normalizer,
             envs, init_functions):
    if cli.eval_episodes % cli.num_envs:
        raise ValueError("eval_episodes must be divisible by num_envs")
    device = cli.device
    task_id = TASK_IDS[cli.env_name]
    successes, rewards, steps, infer_times = [], [], [], []
    solver = None

    for batch_index in range(cli.eval_episodes // cli.num_envs):
        selected = init_functions[
            batch_index * cli.num_envs:(batch_index + 1) * cli.num_envs]
        envs.call_each("run_dill_function", args_list=[(x,) for x in selected])
        obs = envs.reset()
        total_reward = np.zeros(cli.num_envs, dtype=np.float32)
        success = [False] * cli.num_envs
        elapsed_steps = 0
        agent.reset_cache()
        if agent.model["memory"] is not None:
            agent.model["memory"].reset()
            agent.model_ema["memory"].reset()

        while elapsed_steps < 1500:
            started = time.time()
            raw_condition = normalize_observations(obs, dataset, device)
            with torch.no_grad():
                task = torch.full(
                    (cli.num_envs,), task_id, device=device, dtype=torch.long)
                if source_encoder is None:
                    condition = dict(raw_condition)
                    condition["_task_id"] = task
                else:
                    embedding = source_encoder(dict(raw_condition))
                    condition = {"dp_emb": embedding, "_task_id": task}
                prior = torch.zeros(
                    cli.num_envs, int(config.horizon), int(config.action_dim),
                    device=device)
                if str(config.history_method) == "no_history":
                    # The checkpoint uses a Tm=0 DP-PTP decoder.  Reset the
                    # LongMemDDPM container cache on every action chunk so it
                    # behaves as the stateless DP-PTP policy it was trained as.
                    agent.reset_cache()
                normalized_action, _ = agent.sample(
                    prior=prior,
                    n_samples=cli.num_envs,
                    sample_steps=int(config.sample_steps),
                    solver=solver,
                    condition_cfg=condition,
                    w_cfg=1.0,
                    temperature=float(config.temperature),
                    use_ema=True,
                )
            action = action_normalizer.unnormalize(normalized_action)
            action = action.detach().cpu().numpy()
            start = int(config.obs_steps) - 1
            action = action[:, start:start + int(config.action_steps)]
            infer_times.append(time.time() - started)
            obs, reward, _, info = envs.step(action)
            total_reward += np.asarray(reward, dtype=np.float32)
            for env_index in range(cli.num_envs):
                success[env_index] = success[env_index] or any(
                    item["task"] for item in info[env_index]["sc"])
            elapsed_steps += int(config.action_steps)
            if elapsed_steps % 200 == 0:
                print(
                    f"batch={batch_index + 1} steps={elapsed_steps}/1500 "
                    f"successes={sum(success)}/{cli.num_envs}", flush=True)

        if cli.num_video_episodes:
            envs.render()
        successes.extend(success)
        rewards.extend(total_reward.tolist())
        steps.extend([elapsed_steps] * cli.num_envs)
        print(
            f"episodes={batch_index * cli.num_envs + 1}-"
            f"{(batch_index + 1) * cli.num_envs} success={success} ",
            flush=True)

    return {
        "method": str(config.history_method),
        "checkpoint_step": checkpoint_step(cli),
        "env_name": cli.env_name,
        "env_type": cli.env_type,
        "eval_seed": int(cli.seed),
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
        print("cuDNN disabled for this evaluation process", flush=True)
    pathlib.Path(cli.work_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    modality_mapping = collections.defaultdict(list)
    for key, attributes in RAW_SHAPE_META["obs"].items():
        modality_mapping[attributes.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    config = OmegaConf.load(cli.train_config)
    embedding_mode = "embedding_dim" in config
    if embedding_mode and not cli.context_checkpoint:
        raise ValueError("embedding policies require --context-checkpoint")
    dataset = load_raw_dataset(
        cli.dataset_path, int(config.horizon), cli.device,
        proprio_stats=None if embedding_mode else GLOBAL_PROPRIO_STATS)
    source_encoder = (
        build_source_encoder(config, cli.context_checkpoint, cli.device)
        if embedding_mode else None)
    agent, action_normalizer = build_policy(config, cli.checkpoint, cli.device)
    args = env_args(cli, config)
    envs, init_functions = env_factory(config.history_method)(args)
    result = evaluate(
        cli, config, dataset, source_encoder, agent, action_normalizer,
        envs, init_functions)
    result_path = pathlib.Path(cli.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(result_path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
