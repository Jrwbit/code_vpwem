"""Online observation adapter using the same bank and raw-image perception."""
import numpy as np
from momart_boxed_no_history.painting import draw_bbox
from .state import ObjectBank, ObservationStream


class OnlineObjects:
    def __init__(self, config, perceive, num_envs):
        self.config, self.perceive = config, perceive
        m = config.object_memory
        self.streams = [ObservationStream(ObjectBank(int(m.slots), int(m.history),
                                                    int(m.feature_dim), float(m.match_iou)), perceive)
                        for _ in range(num_envs)]
        # Baseline evaluation calls adapter.detector.reset_all at episode boundaries.
        self.detector = self
        self.reset_all()

    def reset_all(self):
        for stream in self.streams:
            stream.reset()
        self.anchor = 0

    def __call__(self, obs):
        values = np.asarray(obs['rgb'])
        if (values.shape != (len(self.streams), 2, 3, 90, 120)
                or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
            raise ValueError('Expected processed batch x 2 x 3 x 90 x 120 RGB in [0,1]')
        images = np.rint(np.moveaxis(values, 2, -1)*255).astype(np.uint8)
        boxed = values.copy()
        snapshots = []
        steps = [max(0, self.anchor-1), self.anchor]
        times = [s*float(self.config.object_memory.period_seconds) for s in steps]
        for b, stream in enumerate(self.streams):
            detections = stream.observe(images[b], steps, times)
            for j, observations in enumerate(detections):
                bowls = [o for o in observations if o.class_id == 0]
                if bowls:
                    outline = draw_bbox(np.zeros_like(images[b,j]),
                                        max(bowls, key=lambda o: o.score).box, True)[...,0] > 0
                    image = np.moveaxis(boxed[b,j], 0, -1)
                    image[outline] = [1., 0., 0.]
            snapshots.append(stream.bank.snapshot(self.config.object_memory.mode))
        result = dict(obs)
        result['rgb'] = boxed
        result.update({'_object_'+key: np.stack([s[key] for s in snapshots]) for key in snapshots[0]})
        self.anchor += int(self.config.action_steps)
        return result
