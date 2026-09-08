import random
import time
import uuid
import os
import json
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
import re
from pathlib import Path
from omegaconf import OmegaConf

from cleandiffuser.env.wrapper import VideoRecordingWrapper


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def auto_resume_training(agent, lr_scheduler, work_dir, gradient_steps):
    """Restore the newest complete model/scheduler pair and return the next step."""
    models_dir = Path(work_dir) / "models"
    if not models_dir.exists():
        return 0
    candidates = []
    for model_path in models_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
        if not match:
            continue
        step = int(match.group(1))
        scheduler_path = models_dir / f"scheduler_{step}.pt"
        if scheduler_path.is_file() and model_path.stat().st_size > 0:
            candidates.append((step, model_path, scheduler_path))
    if not candidates:
        return 0
    step, model_path, scheduler_path = max(candidates, key=lambda item: item[0])
    agent.load(str(model_path))
    lr_scheduler.load_state_dict(
        torch.load(scheduler_path, map_location=agent.device, weights_only=False)
    )
    print(f"Auto-resumed from step {step}: {model_path}", flush=True)
    return min(step + 1, int(gradient_steps) + 1)


def prune_checkpoints(work_dir, keep=3):
    """Keep only the newest complete model/scheduler checkpoint pairs."""
    if keep < 1:
        raise ValueError("keep must be at least 1")

    models_dir = Path(work_dir) / "models"
    if not models_dir.exists():
        return []

    checkpoints = []
    for model_path in models_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
        if not match or model_path.stat().st_size <= 0:
            continue
        step = int(match.group(1))
        scheduler_path = models_dir / f"scheduler_{step}.pt"
        if scheduler_path.is_file() and scheduler_path.stat().st_size > 0:
            checkpoints.append((step, model_path, scheduler_path))

    removed = []
    for step, model_path, scheduler_path in sorted(checkpoints)[:-keep]:
        model_path.unlink()
        scheduler_path.unlink()
        removed.append(step)

    if removed:
        print(f"Pruned checkpoint steps: {removed}; keeping newest {keep}", flush=True)
    return removed


class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik
    
    
class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._video_dir = make_dir(self._log_dir / 'videos')
        self._cfg = cfg

        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=str(uuid.uuid4()),
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def video_init(self, env, enable=False, video_id=""):
        # assert isinstance(env.env, VideoRecordingWrapper)
        if isinstance(env.env, VideoRecordingWrapper):
            video_env = env.env
        else:
            video_env = env
        if enable:
            video_env.video_recoder.stop()
            video_filename = os.path.join(self._video_dir, f"{video_id}_{wv.util.generate_id()}.mp4")
            video_env.file_path = str(video_filename)
        else:
            video_env.file_path = None
            
    def log(self, d, category):
        assert category in ['train', 'inference']
        assert 'step' in d
        def _format_log_value(key, value):
            if key == "step":
                return f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"
            if key == "lr":
                return f"{value:.8e}"
            if "loss" in key:
                return f"{value:.6f}"
            return f"{value:.2f}"

        print(f"[{d['step']}]", " / ".join(f"{k} {_format_log_value(k, v)}" for k, v in d.items()), flush=True)
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = dict()
        for k, v in d.items():
            _d[category + "/" + k] = v
        self._wandb.log(_d, step=d['step'])
        
    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
        agent.save(fp)
        print(f"model_{str(identifier)} saved")

    def finish(self):
        # try:
        #     self.save_agent(agent)
        # except Exception as e:
        #     print(f"Failed to save model: {e}")
        if self._wandb:
            self._wandb.finish()


    
    


    
