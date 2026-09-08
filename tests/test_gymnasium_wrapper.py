from collections import deque

import numpy as np

from cleandiffuser.env.gymnasium_wrapper import MultiStepWrapper


def test_multistep_stacks_emitted_keys_not_stale_observation_space():
    """Eval-only wrapper layers must not reintroduce a stale ``state`` key."""
    wrapper = object.__new__(MultiStepWrapper)
    wrapper.obs = deque([
        {
            "rgb": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "joints": np.zeros((2, 25), dtype=np.float32),
        }
    ])

    stacked = MultiStepWrapper._get_obs(wrapper, n_steps=2)

    assert set(stacked) == {"rgb", "joints"}
    assert stacked["rgb"].shape == (2, 2, 4, 4, 3)
    assert stacked["joints"].shape == (2, 2, 25)
