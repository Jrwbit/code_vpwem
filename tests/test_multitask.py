from cleandiffuser.dataset.multitask import BalancedTaskBatchMux


def test_balanced_task_schedule_is_deterministic():
    loaders = {"a": [1], "b": [2], "c": [3]}
    mux = BalancedTaskBatchMux(loaders, seed=17)
    schedule = [mux.task_for_step(step) for step in range(12)]

    for start in range(0, 12, 3):
        assert set(schedule[start:start + 3]) == set(loaders)
    assert schedule == [
        BalancedTaskBatchMux(loaders, seed=17).task_for_step(step)
        for step in range(12)
    ]


def test_exhausted_task_loader_restarts():
    mux = BalancedTaskBatchMux({"only": ["batch"]}, seed=0)
    assert mux.next(0) == ("only", "batch")
    assert mux.next(1) == ("only", "batch")


def test_empty_loader_has_clear_error():
    mux = BalancedTaskBatchMux({"empty": []}, seed=0)
    try:
        mux.next(0)
    except ValueError as exc:
        assert "produced no full batches" in str(exc)
    else:
        raise AssertionError("empty loader should fail")
