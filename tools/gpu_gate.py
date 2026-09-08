#!/usr/bin/env python3
"""Inspect a shared GPU server and start commands when a GPU becomes free.

No third-party Python packages are required.  Jobs submitted through this tool
are serialized per GPU with file locks; programs started outside the tool are
still detected through nvidia-smi, but cannot participate in that lock.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import pwd
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


STATE_DIR = Path(os.environ.get("GPU_GATE_STATE_DIR", "/tmp/gpu-gate"))


def run_text(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def csv_rows(text: str) -> list[list[str]]:
    return [[cell.strip() for cell in row] for row in csv.reader(text.splitlines())]


def number(value: str) -> int:
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return 0


def gpu_snapshot() -> list[dict]:
    query = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    out = run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not out:
        raise RuntimeError("nvidia-smi 不可用，或当前节点没有 NVIDIA GPU")
    gpus = []
    for row in csv_rows(out):
        if len(row) < 7:
            continue
        gpus.append({
            "index": number(row[0]), "uuid": row[1], "name": row[2],
            "total": number(row[3]), "used": number(row[4]),
            "free": number(row[5]), "util": number(row[6]), "processes": [],
        })

    by_index = {gpu["index"]: gpu for gpu in gpus}
    # query-compute-apps silently omits graphics processes, which can consume
    # several GiB on a workstation. pmon includes both C and G contexts.
    pmon = run_text(["nvidia-smi", "pmon", "-c", "1", "-s", "m"])
    for line in pmon.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = line.split(None, 5)
        if len(row) < 5:
            continue
        gpu_index, pid = number(row[0]), number(row[1])
        if gpu_index not in by_index or pid <= 0:
            continue
        info = process_info(pid)
        info.update({"pid": pid, "type": row[2], "gpu_memory": number(row[3])})
        by_index[gpu_index]["processes"].append(info)
    return gpus


def process_info(pid: int) -> dict:
    result = {"user": "?", "elapsed": "?", "rss_mib": 0, "command": "?"}
    proc = Path("/proc") / str(pid)
    try:
        result["user"] = pwd.getpwuid(proc.stat().st_uid).pw_name
    except (OSError, KeyError):
        pass
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        result["command"] = cmd or (proc / "comm").read_text().strip()
    except OSError:
        pass
    ps = run_text(["ps", "-p", str(pid), "-o", "etimes=,rss="]).strip().split()
    if len(ps) >= 2:
        result["elapsed"] = duration(number(ps[0]))
        result["rss_mib"] = number(ps[1]) // 1024
    return result


def duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def shorten(text: str, width: int = 72) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def load_jobs() -> list[dict]:
    jobs = []
    if not STATE_DIR.exists():
        return jobs
    for path in STATE_DIR.glob("jobs/*.json"):
        try:
            job = json.loads(path.read_text())
            pid = job.get("runner_pid")
            if job.get("state") in {"waiting", "running"} and pid and not Path(f"/proc/{pid}").exists():
                job["state"] = "orphaned"
            jobs.append(job)
        except (OSError, ValueError):
            continue
    return sorted(jobs, key=lambda item: item.get("submitted_at", ""))


def top_ram(limit: int) -> list[list[str]]:
    out = run_text(["ps", "-eo", "user=,pid=,rss=,etimes=,args=", "--sort=-rss"])
    rows = []
    for line in out.splitlines()[:limit]:
        parts = line.strip().split(None, 4)
        if len(parts) == 5:
            rows.append([parts[0], parts[1], f"{number(parts[2]) // 1024} MiB",
                         duration(number(parts[3])), shorten(parts[4])])
    return rows


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    rows_s = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows_s:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows_s:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def cmd_status(args: argparse.Namespace) -> int:
    gpus = gpu_snapshot()
    print("GPU 状态")
    print_table(["GPU", "型号", "显存(用/总)", "空闲", "利用率", "进程数"], [
        [g["index"], shorten(g["name"], 30), f'{g["used"]}/{g["total"]} MiB',
         f'{g["free"]} MiB', f'{g["util"]}%', len(g["processes"])] for g in gpus
    ])
    proc_rows = []
    for gpu in gpus:
        for proc in gpu["processes"]:
            proc_rows.append([gpu["index"], proc.get("type", "C"), proc["user"], proc["pid"],
                              f'{proc["gpu_memory"]} MiB', f'{proc["rss_mib"]} MiB',
                              proc["elapsed"], "未知", shorten(proc["command"])])
    print("\nGPU 进程（C=计算，G=图形；未登记进程的结束时间无法推断）")
    if proc_rows:
        print_table(["GPU", "类型", "用户", "PID", "显存", "RAM", "已运行", "剩余", "命令"], proc_rows)
    else:
        print("无")

    jobs = [j for j in load_jobs() if j.get("state") in {"waiting", "running", "orphaned"}]
    print("\ngpu-gate 队列")
    if jobs:
        rows = []
        for job in jobs:
            end = "未知"
            if job.get("started_epoch") and job.get("estimate_seconds"):
                end = dt.datetime.fromtimestamp(job["started_epoch"] + job["estimate_seconds"]).strftime("%m-%d %H:%M")
            rows.append([job["id"][:8], job.get("user", "?"), job.get("state", "?"),
                         job.get("gpu", "auto"), f'{job.get("min_free_mib", 0)} MiB', end,
                         shorten(job.get("command", ""))])
        print_table(["任务", "用户", "状态", "GPU", "需空闲显存", "预计结束", "命令"], rows)
    else:
        print("无（这里只能看到通过 gpu-gate 提交的候补任务）")

    if args.top_ram:
        print(f"\n全机 RAM 占用前 {args.top_ram} 个进程")
        print_table(["用户", "PID", "RAM", "已运行", "命令"], top_ram(args.top_ram))
    return 0


def parse_timespan(value: str) -> int:
    suffix = value[-1].lower()
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        return int(float(value[:-1]) * factors[suffix]) if suffix in factors else int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("时长格式示例：90m、8h、2d") from exc


def save_job(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    tmp.replace(path)


def ensure_state_dirs() -> None:
    """Create a cross-user spool; sticky dirs protect other users' entries."""
    for path in (STATE_DIR, STATE_DIR / "jobs", STATE_DIR / "locks"):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o1777)
        except PermissionError:
            pass


def choose_gpu(gpus: list[dict], requested: str, min_free: int, max_util: int) -> list[dict]:
    allowed = None if requested == "auto" else {number(x) for x in requested.split(",")}
    choices = [g for g in gpus if (allowed is None or g["index"] in allowed)
               and g["free"] >= min_free and g["util"] <= max_util]
    return sorted(choices, key=lambda g: (g["free"], -g["util"]), reverse=True)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        print("错误：-- 后面必须给出要运行的命令", file=sys.stderr)
        return 2
    ensure_state_dirs()
    job_id = uuid.uuid4().hex
    job_path = STATE_DIR / "jobs" / f"{job_id}.json"
    command_text = shlex.join(args.command)
    job = {
        "id": job_id, "user": pwd.getpwuid(os.getuid()).pw_name, "state": "waiting",
        "gpu": args.gpu, "min_free_mib": args.min_free, "max_util": args.max_util,
        "estimate_seconds": args.estimate, "command": command_text,
        "runner_pid": os.getpid(), "submitted_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    save_job(job_path, job)
    print(f"已加入队列：{job_id[:8]}，等待 GPU={args.gpu}，空闲显存≥{args.min_free} MiB")
    stable_gpu = None
    stable_count = 0
    lock_handle = None
    child = None
    try:
        while True:
            choices = choose_gpu(gpu_snapshot(), args.gpu, args.min_free, args.max_util)
            selected = None
            for gpu in choices:
                lock_path = STATE_DIR / "locks" / f'gpu-{gpu["index"]}.lock'
                # A read-only fd can take flock(2) locks on Linux and remains
                # usable by all users regardless of who created the lock file.
                fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
                candidate = os.fdopen(fd, "r")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected, lock_handle = gpu, candidate
                    break
                except BlockingIOError:
                    candidate.close()
            current = selected["index"] if selected else None
            if current is not None and current == stable_gpu:
                stable_count += 1
            elif current is not None:
                stable_gpu, stable_count = current, 1
            else:
                stable_gpu, stable_count = None, 0
            if selected is not None and stable_count >= args.stable:
                break
            if lock_handle:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
                lock_handle = None
            print(f"[{dt.datetime.now():%H:%M:%S}] 仍在等待；合格 GPU={current}，稳定 {stable_count}/{args.stable}", flush=True)
            time.sleep(args.poll)

        job.update({"state": "running", "gpu": stable_gpu, "started_epoch": time.time(),
                    "started_at": dt.datetime.now().isoformat(timespec="seconds")})
        save_job(job_path, job)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(stable_gpu)
        print(f"[{dt.datetime.now():%F %T}] 启动于物理 GPU {stable_gpu}: {command_text}", flush=True)
        child = subprocess.Popen(args.command, env=env, start_new_session=True)
        job["child_pid"] = child.pid
        save_job(job_path, job)
        rc = child.wait()
        job.update({"state": "finished" if rc == 0 else "failed", "exit_code": rc,
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds")})
        save_job(job_path, job)
        print(f"任务 {job_id[:8]} 结束，退出码 {rc}")
        return rc
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        job["state"] = "cancelled"
        save_job(job_path, job)
        print("\n等待已取消", file=sys.stderr)
        return 130
    finally:
        if lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="共享 GPU 状态查看与空闲自动接入工具")
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", help="查看所有用户的 GPU 进程、队列和 RAM")
    status.add_argument("--top-ram", type=int, default=10, help="显示全机 RAM 前 N 个进程；0 表示关闭")
    status.set_defaults(func=cmd_status)

    launch = sub.add_parser("run", help="资源满足后自动启动命令，并占有本工具的 GPU 锁")
    launch.add_argument("--gpu", default="auto", help="auto、单个编号，或逗号分隔的候选编号")
    launch.add_argument("--min-free", type=int, default=16000, metavar="MiB", help="所需空闲显存")
    launch.add_argument("--max-util", type=int, default=20, metavar="PCT", help="允许的最高 GPU 利用率")
    launch.add_argument("--stable", type=int, default=3, help="连续满足条件的采样次数")
    launch.add_argument("--poll", type=int, default=30, metavar="SEC", help="采样间隔")
    launch.add_argument("--estimate", type=parse_timespan, help="预计运行时长，如 8h（仅展示，不会强杀）")
    launch.add_argument("command", nargs=argparse.REMAINDER, help="-- command arg ...")
    launch.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
