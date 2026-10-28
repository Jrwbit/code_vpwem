"""Queue three 250k boxed No-History evaluations after the existing action-8 queue."""
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

REPO = Path('/home/jrw/code_vpwem_boxed_250k_action4_20261028')
TRAIN = Path('/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_250k_continued_20261027/seed100')
OUTPUT = TRAIN / 'eval_online_combined_blend_250k_action4_seeds100_200_300_20261028'
PREDECESSOR = TRAIN / 'eval_online_combined_blend_250k_seed100_triplicate_20261028'
REFERENCE_REPO = Path('/home/jrw/code_vpwem_boxed_continue_20261027')
BUNDLE = Path('/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json')
PYTHON = '/home/jrw/anaconda3/envs/vpwem/bin/python'
SOURCE_COMMIT = 'bada2c03478689be3854c3efa40afd74314f6640'
ASSIGNMENTS = {100: 3, 200: 4, 300: 6}
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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def environment(gpu=None):
    result = dict(os.environ, PRECISION_CODE_ROOT=str(REPO),
                  PYTHONPATH=f'{REPO}:{REPO}/pipelines', PYTHONDONTWRITEBYTECODE='1',
                  WANDB_MODE='offline', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false')
    if gpu is not None:
        result.update(CUDA_VISIBLE_DEVICES=str(gpu), GIBSON_DEVICE_ID=str(gpu))
    return result


def prepare():
    """Check immutable inputs and load the actual 250k model on CPU before queuing."""
    from omegaconf import OmegaConf
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code'], check=True)
    cfg = OmegaConf.load(TRAIN / 'train_config.yaml')
    assert (cfg.obs_steps, cfg.horizon, cfg.action_steps, cfg.sample_steps) == (2, 9, 8, 50)
    assert int(cfg.seed) == 100
    for task in TASKS:
        assert (Path(cfg.data_root) / (task + '.zarr')).is_dir(), task
    detector_files = ['precision_detector.py', 'detector.py', 'strategies.py']
    detector_hashes = {name: sha(REPO / 'momart_bbox_memory' / name) for name in detector_files}
    assert all(digest == sha(REFERENCE_REPO / 'momart_bbox_memory' / name)
               for name, digest in detector_hashes.items()), 'Detector differs from action-8 reference'
    bundle_hash = sha(BUNDLE)
    assert bundle_hash == 'af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07'
    check_env = environment()
    subprocess.run([PYTHON, '-m', 'momart_boxed_no_history.evaluate', '--help'],
                   env=check_env, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    # Use the same config override as the evaluator: horizon remains nine.
    os.environ.update({k: check_env[k] for k in ('PRECISION_CODE_ROOT', 'HF_HUB_OFFLINE', 'TOKENIZERS_PARALLELISM')})
    sys.path.insert(0, str(REPO))
    import torch
    from momart_boxed_no_history.evaluate import load_normalizers
    from momart_no_history_baseline.model import build_agent
    cfg.device = 'cpu'
    cfg.action_steps = cfg.inference_action_steps = 4
    agent = build_agent(cfg, load_normalizers(cfg, TRAIN, 'cpu'))
    model_path = TRAIN / 'models/model_250000.pt'
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.model.load_state_dict(state['model'], strict=True)
    agent.model_ema.load_state_dict(state['model_ema'], strict=True)
    del state, agent
    manifest = dict(created_at=now(), source_commit=SOURCE_COMMIT, source_root=str(REPO),
        checkpoint=str(model_path), checkpoint_sha256=sha(model_path),
        train_config_sha256=sha(TRAIN / 'train_config.yaml'),
        normalizer_sha256=sha(TRAIN / 'normalizers.npz'),
        detector_bundle=str(BUNDLE), detector_bundle_sha256=bundle_hash,
        detector_source_sha256=detector_hashes, tracker_reset_fix=False,
        reference_protocol='same detector and success rule as existing action-8 triplicate queue',
        method='boxed_no_history', checkpoint_step=250000, training_seed=100,
        prediction_horizon=9, obs_steps=2, training_action_steps=8, execution_action_steps=4,
        sample_steps=50, episodes_per_task=30, num_envs=10, max_environment_steps=1500,
        assignments=ASSIGNMENTS, tasks=TASKS, total_episodes=900,
        predecessor_lock=str(PREDECESSOR / '.supervisor.lock'),
        predecessor_policy='wait for release, regardless of prior experiment outcome',
        strict_cpu_model_and_ema_load=True)
    write_json(OUTPUT / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def read_result(path, seed, task):
    import numpy as np
    value = np.load(path, allow_pickle=True).item()
    expected = dict(method='boxed_no_history', checkpoint_step=250000, eval_seed=seed,
                    execution_action_steps=4, training_action_steps=8, horizon=9, obs_steps=2)
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f'{path}: {key} must be {wanted!r}, got {value.get(key)!r}')
    if value['env_name'] + '_' + value['env_type'] != task:
        raise ValueError('Task identity mismatch')
    success = np.asarray(value['episode_success'])
    if success.shape != (30,) or not np.isin(success, (0, 1)).all():
        raise ValueError('Expected 30 binary episode outcomes')
    if abs(float(success.mean()) - value['mean_success']) > 1e-8:
        raise ValueError('Mean does not match episode outcomes')
    if not np.all(np.asarray(value['episode_steps']) == 1500):
        raise ValueError('Action-4 must run the fixed 1500-step evaluation budget')
    return dict(task=task, successes=int(success.sum()), episodes=30,
                success_rate=float(success.mean()), result_sha256=sha(path))


def summarize():
    runs = {}
    for seed in ASSIGNMENTS:
        rows = []
        for task in TASKS:
            path = OUTPUT / f'seed{seed}' / 'step_250000' / task / 'result.npy'
            if path.exists():
                rows.append(read_result(path, seed, task))
        runs[str(seed)] = dict(completed_tasks=len(rows), total_tasks=len(TASKS), rows=rows)
        if len(rows) == len(TASKS):
            successes = sum(row['successes'] for row in rows)
            runs[str(seed)].update(successes=successes, episodes=300, success_rate=successes / 300)
    result = dict(updated_at=now(), complete=all(x['completed_tasks'] == 10 for x in runs.values()), runs=runs)
    write_json(OUTPUT / 'summary.json', result)
    return result


def worker(seed):
    gpu = ASSIGNMENTS[seed]
    lock_path = Path(f'/home/jrw/vpwem_jobs/momart_gpu_locks/gpu{gpu}.queue.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    root = OUTPUT / f'seed{seed}'
    status = dict(seed=seed, gpu=gpu, pid=os.getpid(), state='waiting_gpu_queue_lock', jobs=[])
    with lock_path.open('a') as lock:
        write_json(root / 'status.json', status)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task in TASKS:
            work = root / 'step_250000' / task
            work.mkdir(parents=True, exist_ok=True)
            result = work / 'result.npy'
            job = dict(task=task, started_at=now())
            status['jobs'].append(job)
            if result.exists():
                job.update(read_result(result, seed, task), state='done', reused=True)
                write_json(root / 'status.json', status)
                continue
            env_name, env_type = task.rsplit('_', 1)
            command = [PYTHON, '-m', 'momart_boxed_no_history.evaluate',
                '--checkpoint', str(TRAIN / 'models/model_250000.pt'),
                '--train-config', str(TRAIN / 'train_config.yaml'),
                '--dataset-path', f'/home/jrw/vpwem_data/momart_zarr/{task}.zarr',
                '--env-name', env_name, '--env-type', env_type,
                '--work-dir', str(work), '--result-path', str(result),
                '--precision-bundle', str(BUNDLE), '--seed', str(seed),
                '--num-envs', '10', '--eval-episodes', '30', '--num-video-episodes', '0',
                '--action-steps', '4']
            log_path = root / 'logs' / (task + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('w') as log:
                process = subprocess.Popen(command, env=environment(gpu), cwd=REPO,
                                           stdout=log, stderr=subprocess.STDOUT)
                job.update(state='running', pid=process.pid, command=command, log=str(log_path))
                status['state'] = 'running'
                write_json(root / 'status.json', status)
                code = process.wait()
            job.update(finished_at=now(), returncode=code)
            if code == 0:
                job.update(read_result(result, seed, task), state='done')
            else:
                job['state'] = 'failed'
            write_json(root / 'status.json', status)
    failed = any(x['state'] != 'done' for x in status['jobs'])
    status.update(state='failed' if failed else 'complete', finished_at=now())
    write_json(root / 'status.json', status)
    return int(failed)


def supervise():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.supervisor.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        for path, key in [(TRAIN / 'models/model_250000.pt', 'checkpoint_sha256'),
                          (TRAIN / 'train_config.yaml', 'train_config_sha256'),
                          (TRAIN / 'normalizers.npz', 'normalizer_sha256'),
                          (BUNDLE, 'detector_bundle_sha256')]:
            assert sha(path) == manifest[key], str(path)
        status = dict(pid=os.getpid(), submitted_at=now(), state='waiting_predecessor',
                      predecessor_lock=str(PREDECESSOR / '.supervisor.lock'), assignments=ASSIGNMENTS)
        write_json(OUTPUT / 'queue_status.json', status)
        summarize()
        print(json.dumps(status), flush=True)
        with (PREDECESSOR / '.supervisor.lock').open('a') as previous:
            fcntl.flock(previous, fcntl.LOCK_SH)
        status.update(state='waiting_gpu', predecessor_released_at=now())
        workers = {}
        for seed, gpu in ASSIGNMENTS.items():
            root = OUTPUT / f'seed{seed}'
            root.mkdir(parents=True, exist_ok=True)
            command = [PYTHON, str(REPO / 'tools/gpu_gate.py'), 'run', '--gpu', str(gpu),
                       '--min-free', '20000', '--max-util', '20', '--stable', '3', '--poll', '10',
                       '--', PYTHON, str(Path(__file__).resolve()), 'worker', '--seed', str(seed)]
            with (root / 'queue.log').open('a') as log:
                workers[seed] = subprocess.Popen(command, env=environment(), stdout=log, stderr=subprocess.STDOUT)
            status.setdefault('worker_pids', {})[seed] = workers[seed].pid
        write_json(OUTPUT / 'queue_status.json', status)
        while any(p.poll() is None for p in workers.values()):
            summarize()
            time.sleep(30)
        summary = summarize()
        codes = {seed: p.returncode for seed, p in workers.items()}
        failed = not summary['complete'] or any(codes.values())
        status.update(state='failed' if failed else 'complete', finished_at=now(), worker_returncodes=codes)
        write_json(OUTPUT / 'queue_status.json', status)
        write_json(OUTPUT / 'completion.json', status)
        return int(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker', 'summary'))
    parser.add_argument('--seed', type=int, choices=tuple(ASSIGNMENTS))
    cli = parser.parse_args()
    if cli.mode == 'prepare':
        prepare()
    elif cli.mode == 'run':
        sys.exit(supervise())
    elif cli.mode == 'worker':
        if cli.seed is None:
            parser.error('worker requires --seed')
        sys.exit(worker(cli.seed))
    else:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
