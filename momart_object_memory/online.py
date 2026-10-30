"""Online observation adapter using the same bank and raw-image perception."""
import numpy as np
from .state import make_bank, ObservationStream


class OnlineObjects:
    def __init__(self, config, perceive, num_envs, main_boxer):
        self.config, self.perceive = config, perceive
        self.main_boxer = main_boxer
        m = config.object_memory
        self.streams = [ObservationStream(make_bank(config), perceive)
                        for _ in range(num_envs)]
        # Baseline evaluation calls adapter.detector.reset_all at episode boundaries.
        self.detector = self
        self.reset_all()

    def reset_all(self):
        for stream in self.streams:
            stream.reset()
        self.main_boxer.detector.reset_all()
        self.anchor = 0

    def __call__(self, obs):
        values = np.asarray(obs['rgb'])
        if (values.shape != (len(self.streams), 2, 3, 90, 120)
                or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
            raise ValueError('Expected processed batch x 2 x 3 x 90 x 120 RGB in [0,1]')
        images = np.rint(np.moveaxis(values, 2, -1)*255).astype(np.uint8)
        snapshots = []
        steps = [max(0, self.anchor-1), self.anchor]
        times = [s*float(self.config.object_memory.period_seconds) for s in steps]
        for b, stream in enumerate(self.streams):
            stream.observe(images[b], steps, times)
            snapshots.append(stream.bank.snapshot(self.config.object_memory.mode))
        # Both branches receive raw RGB; only the baseline boxer paints main RGB.
        result = self.main_boxer(obs)
        result.update({'_object_'+key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]})
        self.anchor += int(self.config.action_steps)
        return result
