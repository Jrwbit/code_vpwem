from pathlib import Path
from unittest.mock import patch
import json
import pytest
from momart_object_memory.prepare_multigpu import partition_datasets, supervise


def test_partition_covers_each_dataset_once_and_balances_load():
    counts = dict(a=110614, b=87120, c=83175, d=69813, e=65586,
                  f=53540, g=52016, h=47253, i=44003, j=41340)
    queues, loads = partition_datasets(counts, [3, 4, 5, 6])
    assigned = [name for names in queues.values() for name in names]
    assert sorted(assigned) == sorted(counts)
    assert sum(loads.values()) == 654460
    assert max(loads.values())-min(loads.values()) < max(counts.values())
    with pytest.raises(ValueError):
        partition_datasets(counts, [3, 3])


def test_workers_only_prepare_and_failure_is_reported(tmp_path):
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            assert command[-1] == '--prepare-only'
            assert command[2] == 'momart_object_memory.prepare_train'
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('3', '4')
        def poll(self):
            return 1
    with patch('momart_object_memory.prepare_multigpu.subprocess.Popen', Child):
        with pytest.raises(RuntimeError, match='Cache workers failed'):
            supervise({3: Path('three.yaml'), 4: Path('four.yaml')}, tmp_path)
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['stage'] == 'failed'
    assert status['policy_training_started'] is False
    assert status['exit_codes'] == {'3': 1, '4': 1}
