"""Recover the incomplete action-8 evaluations on BHR GPUs 0, 1 and 3."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/home/jrw/code_vpwem_boxed_continue_20261027')
TRAIN = Path('/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100')
ROOT = TRAIN / 'eval_online_combined_blend_250k_seed100_triplicate_20261028'
RECOVERY = ROOT / 'recovery_20261029'
BUNDLE = Path('/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json')
PYTHON = '/home/jrw/anaconda3/envs/vpwem/bin/python'
JOBS = {'seed100_repeat3': (100, 0), 'seed200': (200, 1), 'seed300': (300, 3)}
TASKS = [
    'unload_dishwasher_to_dresser_suboptimal', 'unload_dishwasher_to_dresser_expert',
    'table_cleanup_to_sink_suboptimal', 'table_cleanup_to_sink_expert',
    'table_cleanup_to_dishwasher_suboptimal', 'table_cleanup_to_dishwasher_expert',
    'table_setup_from_dishwasher_suboptimal', 'table_setup_from_dishwasher_expert',
    'table_setup_from_dresser_suboptimal', 'table_setup_from_dresser_expert',
]


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_result(path, seed, task):
    import numpy as np
    value = np.load(path, allow_pickle=True).item()
    expected = dict(method='boxed_no_history', checkpoint_step=250000, eval_seed=seed)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Wrong method/checkpoint/seed: {path}')
    if value['env_name'] + '_' + value['env_type'] != task:
        raise ValueError(f'Wrong task: {path}')
    success = np.asarray(value['episode_success'])
    if success.shape != (30,) or not np.isin(success, [0, 1]).all():
        raise ValueError(f'Expected 30 binary episode outcomes: {path}')
    if abs(float(success.mean()) - value['mean_success']) > 1e-8:
        raise ValueError(f'Incorrect success mean: {path}')
    if value.get('execution_action_steps', 8) != 8 or not np.all(np.asarray(value['episode_steps']) == 1504):
        raise ValueError(f'Not the existing action-8 protocol: {path}')
    return dict(task=task, successes=int(success.sum()), episodes=30,
                success_rate=float(success.mean()), sha256=sha(path))


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
    existing = {}; missing = {}
    for label, (seed, gpu) in JOBS.items():
        present = []
        for task in TASKS:
            path = ROOT / label / 'step_250000' / task / 'result.npy'
            if path.exists():
                read_result(path, seed, task)
                existing[str(path)] = sha(path)
                present.append(task)
        missing[label] = [task for task in TASKS if task not in present]
    manifest = dict(created_at=now(), jobs=JOBS, input_sha256=expected,
        source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        source_root=str(REPO), script_sha256=sha(Path(__file__).resolve()),
        existing_results_sha256=existing, missing_tasks=missing,
        new_episode_count=sum(map(len, missing.values())) * 30,
        protocol=dict(checkpoint=250000, action_steps=8, horizon=9, obs_steps=2, num_envs=10,
                      episodes_per_task=30, sample_steps=50, tracker_reset_fix=False),
        provenance='seed100_repeat3 combines original GPU6 valid tasks with GPU0 recovery tasks')
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(RECOVERY / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def completed_rows(label, gpu):
    """Only read newly written NPY files after the evaluator records DONE/SKIP."""
    path = ROOT / label / f'gpu{gpu}_status.tsv'
    tasks = set()
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split('\t')
            if len(fields) >= 4 and fields[1] in ('DONE', 'SKIP') and fields[2] == '250000':
                tasks.add(fields[3])
    return tasks


def summarize(final=False):
    manifest = json.loads((RECOVERY / 'manifest.json').read_text())
    runs = {}
    for label, (seed, gpu) in JOBS.items():
        marked_done = completed_rows(label, gpu)
        rows = []
        for task in TASKS:
            path = ROOT / label / 'step_250000' / task / 'result.npy'
            if path.exists() and (final or task in marked_done or str(path) in manifest['existing_results_sha256']):
                rows.append(read_result(path, seed, task))
        runs[label] = dict(seed=seed, recovery_gpu=gpu, completed_tasks=len(rows), total_tasks=10, rows=rows)
        if len(rows) == 10:
            successes = sum(row['successes'] for row in rows)
            runs[label].update(successes=successes, episodes=300, success_rate=successes / 300)
    for path, digest in manifest['existing_results_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Previously valid result was changed: {path}')
    result = dict(updated_at=now(), complete=all(r['completed_tasks'] == 10 for r in runs.values()), runs=runs)
    write_json(RECOVERY / 'summary.json', result)
    return result


def supervise():
    manifest = json.loads((RECOVERY / 'manifest.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path(__file__).resolve()) == manifest['script_sha256']
    with (ROOT / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), submitted_at=now(), state='dispatched', jobs={})
        workers = {}
        for label, (seed, gpu) in JOBS.items():
            output = ROOT / label
            output.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, MOMART_REPO=str(REPO), MOMART_RUN_ROOT=str(TRAIN),
                MOMART_EVAL_ROOT=str(output), EVAL_METHOD='boxed_no_history', EVAL_STEPS='250000',
                GPU=str(gpu), EVAL_SEED=str(seed), EVAL_EPISODES='30', NUM_ENVS='10',
                PRECISION_CODE_ROOT=str(REPO), PRECISION_BUNDLE=str(BUNDLE), VPWEM_PYTHON=PYTHON,
                OPENBLAS_NUM_THREADS='1', MOMART_GPU_LOCK_WAIT='1')
            command = [PYTHON, str(REPO / 'tools/gpu_gate.py'), 'run', '--gpu', str(gpu),
                '--min-free', '22000', '--max-util', '20', '--stable', '3', '--poll', '10',
                '--', 'bash', str(REPO / 'tools/run_momart_boxed_checkpoint_eval_bhr.sh')]
            with (RECOVERY / f'{label}_queue.log').open('a') as log:
                workers[label] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            status['jobs'][label] = dict(seed=seed, gpu=gpu, runner_pid=workers[label].pid,
                                        output=str(output), command=command)
        write_json(RECOVERY / 'queue_status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        while any(p.poll() is None for p in workers.values()):
            summarize()
            time.sleep(30)
        summary = summarize(final=True)
        codes = {label: p.returncode for label, p in workers.items()}
        failed = not summary['complete'] or any(codes.values())
        status.update(state='failed' if failed else 'complete', finished_at=now(), returncodes=codes)
        write_json(RECOVERY / 'queue_status.json', status)
        write_json(RECOVERY / 'completion.json', status)
        return int(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'summary'))
    cli = parser.parse_args()
    if cli.mode == 'prepare':
        prepare()
    elif cli.mode == 'run':
        sys.exit(supervise())
    else:
        print(json.dumps(summarize(final=True), ensure_ascii=False, indent=2))
