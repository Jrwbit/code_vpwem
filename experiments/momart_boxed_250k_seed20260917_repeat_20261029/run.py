"""Repeat the original 250k boxed evaluation on physical GPU 4."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path('/home/jrw/code_vpwem_boxed_continue_20261027')
TRAIN = Path('/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100')
ROOT = TRAIN / 'eval_online_combined_blend_250k_seed20260917_repeat_gpu4_20261029'
REFERENCE = TRAIN / 'eval_online_combined_blend_250k'
BUNDLE = Path('/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json')
PYTHON = '/home/jrw/anaconda3/envs/vpwem/bin/python'


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def prepare():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(TRAIN / 'train_config.yaml')
    assert (cfg.horizon, cfg.obs_steps, cfg.action_steps, cfg.sample_steps) == (9, 2, 8, 50)
    expected = {
        str(TRAIN / 'models/model_250000.pt'): '00ccf12e6f4f9dec1965c6139da9da385cf3792f0e2a23ca9b2945cfafc5b203',
        str(TRAIN / 'train_config.yaml'): 'ffb26097aa459dbbbc91514313c4563cdd1cd9ccb9b3db41c69f78650153a4bb',
        str(BUNDLE): 'af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07',
    }
    for path, digest in expected.items():
        assert sha(Path(path)) == digest, path
    sources = ['momart_boxed_no_history/evaluate.py', 'momart_bbox_memory/precision_detector.py',
               'momart_bbox_memory/detector.py', 'momart_bbox_memory/strategies.py',
               'tools/run_momart_boxed_checkpoint_eval_bhr.sh']
    manifest = dict(created_at=now(), gpu=4, eval_seed=20260917, training_seed=100,
                    checkpoint_step=250000, prediction_horizon=9, action_steps=8,
                    obs_steps=2, sample_steps=50, num_envs=10, episodes_per_task=30,
                    total_episodes=300, tracker_reset_fix=False, reference=str(REFERENCE),
                    input_sha256=expected, source_sha256={s: sha(REPO / s) for s in sources},
                    source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    script_sha256=sha(Path(__file__).resolve()))
    ROOT.mkdir(parents=True, exist_ok=True)
    # An existing run must be inspected, not silently overwritten or reused.
    with (ROOT / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    write('queue_status.json', dict(state='prepared', updated_at=now(), gpu=4))
    print(json.dumps(manifest, ensure_ascii=False))


def summarize():
    import numpy as np
    reference = sorted((REFERENCE / 'step_250000').glob('*/result.npy'))
    assert len(reference) == 10
    success = total = matched = 0
    for old in reference:
        path = ROOT / 'step_250000' / old.parent.name / 'result.npy'
        value = np.load(path, allow_pickle=True).item()
        assert value['method'] == 'boxed_no_history'
        assert value['checkpoint_step'] == 250000 and value['eval_seed'] == 20260917
        assert value['env_name'] + '_' + value['env_type'] == old.parent.name
        flags = np.asarray(value['episode_success'])
        assert flags.shape == (30,) and np.isin(flags, [0, 1]).all()
        assert np.isclose(flags.mean(), value['mean_success'])
        previous = np.load(old, allow_pickle=True).item()
        assert previous['eval_seed'] == 20260917 and previous['checkpoint_step'] == 250000
        matched += int(np.sum(flags == np.asarray(previous['episode_success'])))
        success += int(flags.sum())
        total += len(flags)
    result = dict(completed_at=now(), tasks=10, episodes=total, successes=success,
                  success_rate=success / total, reference_successes=194,
                  identical_episode_flags=matched, all_flags_identical=matched == total)
    write('summary.json', result)
    return result


def run():
    with (ROOT / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for path, digest in manifest['input_sha256'].items():
            assert sha(Path(path)) == digest, path
        for path, digest in manifest['source_sha256'].items():
            assert sha(REPO / path) == digest, path
        assert sha(Path(__file__).resolve()) == manifest['script_sha256']
        env = dict(os.environ, MOMART_REPO=str(REPO), MOMART_RUN_ROOT=str(TRAIN),
                   MOMART_EVAL_ROOT=str(ROOT), EVAL_METHOD='boxed_no_history', EVAL_STEPS='250000',
                   GPU='4', EVAL_SEED='20260917', EVAL_EPISODES='30', NUM_ENVS='10',
                   PRECISION_CODE_ROOT=str(REPO), PRECISION_BUNDLE=str(BUNDLE), VPWEM_PYTHON=PYTHON,
                   OPENBLAS_NUM_THREADS='1', MOMART_GPU_LOCK_WAIT='1')
        command = [PYTHON, str(REPO / 'tools/gpu_gate.py'), 'run', '--gpu', '4',
                   '--min-free', '22000', '--max-util', '20', '--stable', '3', '--poll', '10',
                   '--', 'bash', str(REPO / 'tools/run_momart_boxed_checkpoint_eval_bhr.sh')]
        status = dict(state='waiting_for_gpu_or_running', pid=os.getpid(), gpu=4,
                      updated_at=now(), command=command)
        write('queue_status.json', status)
        code = subprocess.call(command, env=env)
        status.update(state='failed' if code else 'complete', returncode=code, updated_at=now())
        if code == 0:
            try:
                summarize()
            except Exception:
                status.update(state='validation_failed')
                write('queue_status.json', status)
                raise
        write('queue_status.json', status)
        write('completion.json', status)
        return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'summary'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        sys.exit(run())
    else:
        print(json.dumps(summarize(), ensure_ascii=False))
