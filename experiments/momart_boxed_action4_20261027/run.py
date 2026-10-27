"""Run the fixed 150k boxed No-History action-4 experiment on BHR GPUs 3/5/6."""
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

REPO = Path('/home/jrw/code_vpwem_boxed_action4_20261027')
OUTPUT = Path('/home/jrw/vpwem_outputs/momart_boxed_no_history_150k_action4_20261027_gpu356')
TRAIN = Path('/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100')
BASELINE = Path('/home/jrw/vpwem_outputs/momart_boxed_online_150k_combined_blend_20261026_gpu1/boxed_no_history/step_150000')
PRECISION = Path('/home/jrw/code_vpwem_bbox_online_20261020')
BUNDLE = Path('/home/jrw/vpwem_outputs/bbox_strategies_20261020/comparison_v2/detector_bundle.json')
PYTHON = '/home/jrw/anaconda3/envs/vpwem/bin/python'
TASKS = [
    'unload_dishwasher_to_dresser_suboptimal',
    'unload_dishwasher_to_dresser_expert',
    'table_cleanup_to_sink_suboptimal',
    'table_cleanup_to_sink_expert',
    'table_cleanup_to_dishwasher_suboptimal',
    'table_cleanup_to_dishwasher_expert',
    'table_setup_from_dishwasher_suboptimal',
    'table_setup_from_dishwasher_expert',
    'table_setup_from_dresser_suboptimal',
    'table_setup_from_dresser_expert',
]
ASSIGNMENTS = {gpu: TASKS[offset::3] for offset, gpu in enumerate((3, 5, 6))}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    """Patch only an isolated checkout of the exact baseline evaluator."""
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == 'dbf6779ac90c7a19b7543a1eb3d5509ef2253f13'
    expected = {
        TRAIN / 'models/model_150000.pt': 'b33b859502decba0c734734bedf91df2b3ba28a35c17b7eeff7eae25989a59ab',
        BUNDLE: 'af62860467a1c95d2e60a9df5d89ee1b9001116cb45c1aa7a285c1810f6f7c07',
    }
    for path, digest in expected.items():
        assert sha(path) == digest, str(path)
    evaluator = REPO / 'momart_boxed_no_history/evaluate.py'
    source = evaluator.read_text()
    replacements = [
        ('    return parser.parse_args()',
         '    parser.add_argument("--action-steps", type=int, default=None)\n    return parser.parse_args()'),
        ('    config = OmegaConf.load(cli.train_config)',
         '    config = OmegaConf.load(cli.train_config)\n'
         '    config.training_action_steps = int(config.action_steps)\n'
         '    if cli.action_steps is not None:\n'
         '        if not 1 <= cli.action_steps <= int(config.horizon) - int(config.obs_steps) + 1:\n'
         '            raise ValueError("action steps exceed available future actions")\n'
         '        config.action_steps = cli.action_steps\n'
         '    print(f"execution_action_steps={config.action_steps} horizon={config.horizon} obs_steps={config.obs_steps}", flush=True)'),
        ('        "eval_seed": cli.seed,',
         '        "eval_seed": cli.seed,\n'
         '        "execution_action_steps": int(config.action_steps),\n'
         '        "training_action_steps": int(config.training_action_steps),\n'
         '        "horizon": int(config.horizon), "obs_steps": int(config.obs_steps),'),
        ('    np.save(result_path, result)',
         '    temporary = result_path.with_suffix(result_path.suffix + ".tmp")\n'
         '    with temporary.open("wb") as stream:\n'
         '        np.save(stream, result)\n'
         '    temporary.replace(result_path)'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    compile(source, str(evaluator), 'exec')
    evaluator.write_text(source)
    write_json(OUTPUT / 'manifest.json', dict(
        created_at=now(), checkpoint=str(TRAIN / 'models/model_150000.pt'),
        checkpoint_sha256=expected[TRAIN / 'models/model_150000.pt'],
        train_config_sha256=sha(TRAIN / 'train_config.yaml'),
        detector_bundle=str(BUNDLE), detector_bundle_sha256=expected[BUNDLE],
        source_commit='dbf6779ac90c7a19b7543a1eb3d5509ef2253f13',
        physical_gpu_assignments=ASSIGNMENTS, action_steps=4, horizon=9, obs_steps=2,
        eval_seed=20260917, episodes_per_task=30, num_envs=10, sample_steps=50,
        max_environment_steps=1500, baseline_successes=178, baseline_episodes=300,
        baseline_root=str(BASELINE), tracker_update_frequency_changes=True,
        success_rule='unchanged: OR of returned info sc task flags',
    ))


def read_result(path, task, action4=True):
    import numpy as np
    result = np.load(path, allow_pickle=True).item()
    success = np.asarray(result['episode_success'])
    assert result['method'] == 'boxed_no_history'
    assert result['checkpoint_step'] == 150000 and result['eval_seed'] == 20260917
    assert result['env_name'] + '_' + result['env_type'] == task
    assert len(success) == 30 and np.isin(success, (0, 1)).all()
    assert abs(float(success.mean()) - result['mean_success']) < 1e-8
    if action4:
        assert result['execution_action_steps'] == 4
        assert result['training_action_steps'] == 8
        assert result['horizon'] == 9 and result['obs_steps'] == 2
        assert np.all(np.asarray(result['episode_steps']) == 1500)
    return dict(successes=int(success.sum()), episodes=30,
                success_rate=float(success.mean()),
                mean_infer_time=float(result['mean_infer_time']),
                result_sha256=sha(path))


def worker(gpu):
    lock_path = Path(f'/home/jrw/vpwem_jobs/momart_gpu_locks/gpu{gpu}.queue.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GIBSON_DEVICE_ID=str(gpu),
               PRECISION_CODE_ROOT=str(PRECISION),
               PYTHONPATH=f'{REPO}:{REPO}/pipelines:{PRECISION}',
               PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='offline', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false')
    status = dict(gpu=gpu, pid=os.getpid(), started_at=now(), jobs=[])
    status_path = OUTPUT / f'gpu{gpu}_status.json'
    for task in ASSIGNMENTS[gpu]:
        work = OUTPUT / 'step_150000' / task
        work.mkdir(parents=True, exist_ok=True)
        result = work / 'result.npy'
        job = dict(task=task, state='starting', started_at=now())
        status['jobs'].append(job)
        if result.exists():
            job.update(read_result(result, task), state='done', reused=True)
            write_json(status_path, status)
            continue
        env_name, env_type = task.rsplit('_', 1)
        command = [PYTHON, '-m', 'momart_boxed_no_history.evaluate',
                   '--checkpoint', str(TRAIN / 'models/model_150000.pt'),
                   '--train-config', str(TRAIN / 'train_config.yaml'),
                   '--dataset-path', f'/home/jrw/vpwem_data/momart_zarr/{task}.zarr',
                   '--env-name', env_name, '--env-type', env_type,
                   '--work-dir', str(work), '--result-path', str(result),
                   '--precision-bundle', str(BUNDLE), '--seed', '20260917',
                   '--num-envs', '10', '--eval-episodes', '30',
                   '--num-video-episodes', '0', '--action-steps', '4']
        log_path = OUTPUT / 'logs' / f'gpu{gpu}_{task}.log'
        started = time.monotonic()
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            job.update(state='running', pid=process.pid, command=command, log=str(log_path))
            write_json(status_path, status)
            code = process.wait()
        job.update(returncode=code, wall_seconds=time.monotonic() - started, finished_at=now())
        if code == 0:
            job.update(read_result(result, task), state='done')
        else:
            job['state'] = 'failed'
        write_json(status_path, status)
    status['finished_at'] = now()
    write_json(status_path, status)
    return int(any(job['state'] != 'done' for job in status['jobs']))


def summarize():
    rows = []
    for task in TASKS:
        result = OUTPUT / 'step_150000' / task / 'result.npy'
        if result.exists():
            row = dict(task=task, action4=read_result(result, task),
                       action8=read_result(BASELINE / task / 'result.npy', task, False))
            rows.append(row)
    summary = dict(updated_at=now(), complete=len(rows) == len(TASKS),
                   completed_tasks=len(rows), total_tasks=len(TASKS), rows=rows)
    if summary['complete']:
        current = sum(row['action4']['successes'] for row in rows)
        baseline = sum(row['action8']['successes'] for row in rows)
        assert baseline == 178
        summary.update(action4_successes=current, action8_successes=baseline,
                       episodes=300, action4_success_rate=current / 300,
                       difference_pp=(current - baseline) / 3)
    write_json(OUTPUT / 'summary.json', summary)
    return summary


def supervise():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'logs').mkdir(exist_ok=True)
    lock = (OUTPUT / '.supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    manifest.update(evaluation_commit=subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        launched_at=now(), supervisor_pid=os.getpid())
    assert sha(TRAIN / 'train_config.yaml') == manifest['train_config_sha256']
    write_json(OUTPUT / 'manifest.json', manifest)
    workers = {}
    for gpu in ASSIGNMENTS:
        with (OUTPUT / 'logs' / f'gpu{gpu}_worker.log').open('a') as log:
            workers[gpu] = subprocess.Popen([PYTHON, __file__, 'worker', '--gpu', str(gpu)],
                                            stdout=log, stderr=subprocess.STDOUT)
    while any(process.poll() is None for process in workers.values()):
        summarize()
        time.sleep(30)
    summary = summarize()
    write_json(OUTPUT / 'completion.json', dict(finished_at=now(),
        worker_returncodes={gpu: process.returncode for gpu, process in workers.items()},
        complete=summary['complete']))
    return int(not summary['complete'] or any(p.returncode for p in workers.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker', 'summary'))
    parser.add_argument('--gpu', type=int, choices=(3, 5, 6))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        sys.exit(supervise())
    elif args.mode == 'worker':
        if args.gpu is None:
            parser.error('worker requires --gpu')
        sys.exit(worker(args.gpu))
    else:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
