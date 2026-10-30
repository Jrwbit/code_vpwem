"""DP-PTP MoMaRT online environments for the No-History baseline."""

import collections
import os
import pathlib
import time

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "450")
os.environ.setdefault("MESA_GLES_VERSION_OVERRIDE", "4.5")

import dill

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from .metadata import get_env_metadata

from cleandiffuser.env.async_vector_env import AsyncVectorEnv
from cleandiffuser.env.robomimic.momart_image_wrapper import MomartImageWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.env.wrapper import MultiStepWrapper, VideoRecordingWrapper


def make_async_envs(args):
    """Match the short-observation window used by the completed 1M evaluation."""
    env_meta = get_env_metadata(args.dataset_path)

    def create_env(enable_render):
        mapping = collections.defaultdict(list)
        for key, attr in args.shape_meta["obs"].items():
            mapping[attr.get("type", "low_dim")].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(mapping)
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=enable_render,
            use_image_obs=enable_render)
        return env

    def wrap_env(enable_render):
        env = create_env(enable_render)
        if enable_render:
            env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                MomartImageWrapper(
                    env=env, shape_meta=args.shape_meta,
                    init_state=None, render_obs_key="rgb"),
                video_recoder=VideoRecorder.create_h264(
                    fps=20, codec="h264", input_pix_fmt="rgb24",
                    crf=22, thread_type="FRAME", thread_count=1),
                file_path=None, steps_per_render=5,
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps,
        )

    def env_fn():
        return wrap_env(True)

    def dummy_env_fn():
        # AsyncVectorEnv creates spaces before the workers start. Avoid opening
        # an OpenGL context in the parent process.
        return wrap_env(False)

    initializers = []
    for episode in range(args.eval_episodes):
        def init_fn(env, episode=episode):
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if args.save_video:
                filename = pathlib.Path(args.work_dir) / "videos" / f"{time.time()}_{episode}.mp4"
                filename.parent.mkdir(parents=True, exist_ok=True)
                env.env.file_path = filename
        initializers.append(dill.dumps(init_fn))

    envs = AsyncVectorEnv([env_fn] * args.num_envs, dummy_env_fn=dummy_env_fn)
    envs.seed(int(args.seed))
    return envs, initializers
