"""Utilities for balanced multi-task offline training."""

import random


class BalancedTaskBatchMux:
    """Draw homogeneous batches from several task loaders with equal exposure.

    Keeping every batch task-homogeneous lets each dataset retain its own
    collator and action/observation normalizers. Task order is shuffled once
    per balanced cycle and is a pure function of the seed and optimizer step,
    so checkpoint resume preserves the task schedule.
    """

    def __init__(self, loaders, seed=0):
        if not loaders:
            raise ValueError("BalancedTaskBatchMux requires at least one loader")
        self.loaders = dict(loaders)
        self.task_names = tuple(self.loaders)
        self.seed = int(seed)
        self._iterators = {}

    def task_for_step(self, step):
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        cycle, offset = divmod(step, len(self.task_names))
        order = list(self.task_names)
        random.Random(self.seed + cycle).shuffle(order)
        return order[offset]

    def next(self, step):
        task_name = self.task_for_step(step)
        iterator = self._iterators.get(task_name)
        if iterator is None:
            iterator = iter(self.loaders[task_name])
            self._iterators[task_name] = iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self.loaders[task_name])
            self._iterators[task_name] = iterator
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"Task loader {task_name!r} produced no full batches"
                ) from exc
        return task_name, batch
