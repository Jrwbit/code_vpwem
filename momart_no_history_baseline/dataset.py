"""MoMaRT samples containing only the observations used by No-History."""

import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.utils.robomimic_dataset_utils import LongMemSequenceDataset


class NoHistorySequenceDataset(LongMemSequenceDataset):
    """Keep the original action horizon and sampler, but read only To observations.

    The parent dataset's get_item calls this method for observations. Its
    standard implementation also reads and processes up to 16 history frames.
    No-History never consumes those frames, so this override avoids the HDF5
    reads, collation, and GPU transfers without changing action targets.
    """

    def __init__(self, *args, obs_steps, **kwargs):
        self.obs_steps = int(obs_steps)
        if self.obs_steps < 1 or self.obs_steps > int(kwargs["seq_length"]):
            raise ValueError("obs_steps must be between 1 and seq_length")
        super().__init__(*args, **kwargs)

    def get_obs_sequence_from_demo(
        self, demo_id, index_in_demo, keys, num_frames_to_stack=0,
        seq_length=1, prefix="obs",
    ):
        obs, _ = self.get_sequence_from_demo(
            demo_id, index_in_demo,
            keys=tuple(f"{prefix}/{key}" for key in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=min(seq_length, self.obs_steps),
        )
        return ObsUtils.process_obs_dict({
            key: obs[f"{prefix}/{key}"] for key in keys
        })
