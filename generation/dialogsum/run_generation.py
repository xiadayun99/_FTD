#!/usr/bin/env python3
"""Run the FTD DialogSum pilot or confirmation in a separate output directory."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import uuid

import generation_protocol as gp

HERE = Path(__file__).resolve().parent
BASE_DEFAULT = Path("assets/Qwen3-8B")
REFERENCE_DEFAULT = Path("assets/reference_run")
OUTPUT_DEFAULT = Path("runs/dialogsum")
PREPARE = [
    dict(key="data", phase="data"),
    dict(key="gpu_probe", phase="gpu_probe"),
    dict(key=f"train_{gp.TASK}", phase="train"),
    dict(key=f"merge_{gp.TASK}", phase="merge"),
    dict(key="build_direct", phase="build", method="direct"),
    dict(key="build_shared", phase="build", method="shared"),
]
WEIGHT_EXTENSIONS = (".safetensors", ".safetensors.partial", ".pt", ".pth", ".bin")


def resolve_inputs(args):
    spec = gp.read_json(HERE / "paper_assets.json")
    model = args.model_dir.resolve()
    source = args.reference_run.resolve()
    asset_hashes = {model / name: digest for name, digest in spec["model_files_sha256"].items()}
    reference_paths = (
        ["artifacts/references/metadata.json"] +
        [f"artifacts/references/{task}.{extension}" for task in ("base", *gp.PEERS)
         for extension in ("json", "safetensors")] +
        [f"trained/lora/{task}/{filename}" for task in gp.PEERS
         for filename in ("adapter_config.json", "adapter_model.safetensors")]
    )
    asset_hashes.update({source / name: spec["reference_run_files_sha256"][name] for name in reference_paths})
    absent = [str(path) for path in asset_hashes if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing paper assets:\n  " + "\n  ".join(absent))
    for path, digest in asset_hashes.items():
        print(f"CHECK_ASSET: {path}", flush=True)
        if gp.sha(path) != digest:
            raise ValueError(f"Wrong paper asset hash: {path}")
    return dict(protocol=gp.PROTOCOL_VERSION, model_dir=str(model), reference_run=str(source),
        release_sha256=gp.sha(HERE / "MANIFEST_SHA256.json"),
        pinned_assets={str(path): digest for path, digest in asset_hashes.items()})


def check_release():
    manifest = gp.read_json(HERE / "MANIFEST_SHA256.json")
    if not manifest:
        raise ValueError("Empty release manifest")
    for name, digest in manifest.items():
        file = HERE / name
        if (Path(name).is_absolute() or not file.resolve().is_relative_to(HERE)
                or not file.is_file() or file.is_symlink() or gp.sha(file) != digest):
            raise ValueError(f"Release file changed: {name}")


def link_paper_assets(root, config):
    source = Path(config["reference_run"])
    links = [f"trained/lora/{task}" for task in gp.PEERS]
    links += [f"artifacts/references/{task}.{ext}" for task in ("base", *gp.PEERS)
              for ext in ("json", "safetensors")]
    for relative in links:
        path, target = root / relative, source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise ValueError(f"Input link changed: {path}")
        elif path.exists():
            raise FileExistsError(f"Refusing to replace existing path: {path}")
        else:
            path.symlink_to(target, target_is_directory=target.is_dir())


def outputs(root, job):
    phase = job["phase"]
    if phase == "data":
        return [root / "inputs"]
    if phase == "gpu_probe":
        return [root / "runtime_contract.json", root / "gpu_probe.json"]
    if phase == "train":
        return [root / "trained/lora" / gp.TASK, root / "training.jsonl"]
    if phase == "merge":
        ref = root / "artifacts/references"
        return [ref / f"{gp.TASK}.safetensors", ref / f"{gp.TASK}.json", ref / "metadata.json"]
    if phase == "build":
        bank = "patches/direct" if job["method"] == "direct" else "shared_target"
        return [root / "artifacts" / bank]
    if phase == "trace":
        return [root / "stages" / job["key"]]
    raise ValueError(f"Unknown job {job}")


def inventory(root, paths):
    result = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        if not files or any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
            raise ValueError(f"Missing/invalid phase outputs: {path}")
        for file in files:
            result[str(file.relative_to(root))] = gp.sha(file)
    return result


def incomplete(root, job):
    backup = root / "interrupted" / f"{job['key']}_{uuid.uuid4().hex[:8]}"
    for output in outputs(root, job):
        partial = output.with_name(output.name + ".partial")
        for item in (output, partial):
            if not item.exists():
                continue
            if item.is_symlink() or not item.resolve().is_relative_to(root):
                raise ValueError(f"Unexpected interrupted output: {item}")
            # Free only uncommitted tensor payloads from this known phase.
            # Diagnostic files are retained. Paper assets are never written.
            files = [item] if item.is_file() else [p for p in item.rglob("*") if p.is_file()]
            for file in files:
                if file.is_symlink() or not file.resolve().is_relative_to(root):
                    raise ValueError(f"Unexpected phase file: {file}")
                if file.name.endswith(WEIGHT_EXTENSIONS):
                    file.unlink()
            if item.exists():
                dest = backup / item.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                item.rename(dest)


def wait_gpu(seconds):
    import pynvml as nv

    nv.nvmlInit()
    try:
        if nv.nvmlDeviceGetCount() != 1:
            raise ValueError("Expose exactly one NVIDIA GPU to CUDA and NVML")
        handle = nv.nvmlDeviceGetHandleByIndex(0)
        deadline = time.monotonic() + seconds
        since, last = None, 0.
        while time.monotonic() < deadline:
            now = time.monotonic()
            free = nv.nvmlDeviceGetMemoryInfo(handle).free
            fields = gp.read("/proc/meminfo").splitlines()
            host = int(next(line.split()[1] for line in fields if line.startswith("MemAvailable:"))) * 1024
            pids = [proc.pid for proc in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
            if not pids and free >= gp.MIN_GPU_FREE and host >= gp.MIN_HOST_AVAILABLE:
                since = now if since is None else since
                if now - since >= 30:
                    print("GPU_IDLE_CONFIRMED", flush=True)
                    return
            else:
                since = None
            if now - last >= 30:
                print(f"WAITING_FOR_IDLE_GPU: pids={pids}, free GPU={free/gp.GIB:.1f} GiB, "
                      f"host available={host/gp.GIB:.1f} GiB", flush=True)
                last = now
            time.sleep(2)
    finally:
        nv.nvmlShutdown()
    raise TimeoutError("Timed out waiting for 30 s of GPU isolation and memory headroom")


def worker_env():
    env = dict(os.environ)
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONFAULTHANDLER="1",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def run_worker(root, job, data_dir, timeout, lock_fd):
    attempt = root / "attempts" / f"{job['key']}_{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    gp.write_json(attempt / "job.json", job)
    gp.write_json(attempt / "settings.json", dict(data_dir=str(data_dir.resolve()) if data_dir else None))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker", str(root), str(attempt)]
    log = attempt / "worker.log"
    started = time.monotonic()
    print(f"PHASE: {job['key']} | {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=worker_env(), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lock_fd,))
        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise TimeoutError(f"Worker time limit exceeded: {job['key']}")
                try:
                    code = process.wait(timeout=min(30, timeout-elapsed))
                    break
                except subprocess.TimeoutExpired:
                    print(f"ACTIVE: {job['key']} elapsed={elapsed+30:.0f}s", flush=True)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
    gp.write_json(attempt / "exit.json", dict(code=code, elapsed_s=time.monotonic()-started))
    return code, attempt


def phase(root, job, args, lock_fd):
    path = root / "completed" / f"{job['key']}.json"
    if path.exists():
        marker = gp.read_json(path)
        if marker["job"] != job or marker["outputs"] != inventory(root, outputs(root, job)):
            raise ValueError(f"Completed phase changed: {job['key']}")
        print(f"REUSE: {job['key']}", flush=True)
        return
    for attempt_index in range(3):
        incomplete(root, job)
        if job["phase"] != "data":
            wait_gpu(args.idle_timeout)
        code, attempt = run_worker(root, job, args.data_dir, args.phase_timeout, lock_fd)
        if code == 0:
            gp.write_json(path, dict(job=job, outputs=inventory(root, outputs(root, job)),
                                    attempt=str(attempt.relative_to(root))))
            return
        log = gp.read(attempt / "worker.log")
        tail = "\n".join(log.splitlines()[-22:])
        if attempt_index < 2 and job["phase"] != "data" and (
            "Observed foreign GPU device users:" in log or "GPU_ISOLATION_FAILED: observed foreign" in log
        ):
            print(f"GPU_INTERFERENCE_RETRY: {job['key']} attempt={attempt_index+1} | {attempt}", flush=True)
            continue
        raise RuntimeError(f"Worker failed ({code}) at {job['key']}\n{tail}\nLog: {attempt / 'worker.log'}")


def archive(root, scope):
    suffix = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    path = root / f"{gp.ARCHIVE_PREFIX}{scope}_{suffix}.tar.gz"
    candidates = [root / filename for filename in (
        "status.json", "run_config.json", "runtime_contract.json", "gpu_probe.json",
        "pilot_summary.json", "confirm_summary.json", "training.jsonl", "inputs/cohort.json",
        "inputs/workload.json", f"trained/lora/{gp.TASK}/training_contract.json",
        f"artifacts/references/{gp.TASK}.json", "artifacts/references/metadata.json",
        "artifacts/patches/direct/build_report.json", "artifacts/shared_target/build_report.json",
    )]
    for subdir in ("stages", "attempts", "interrupted", "completed"):
        candidates.extend(p for p in (root / subdir).rglob("*") if p.is_file())
    with tarfile.open(path, "w:gz", compresslevel=3) as stream:
        for file in sorted(set(candidates)):
            if (file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root)
                    and not file.name.endswith(WEIGHT_EXTENSIONS)):
                stream.add(file, arcname=f"{root.name}/{file.relative_to(root)}", recursive=False)
        for filename in ("START_HERE_zh.md", "MANIFEST_SHA256.json", "generation_protocol.py"):
            stream.add(HERE / filename, arcname=f"{root.name}/release/{filename}", recursive=False)
    print(f"SEND_THIS_ARCHIVE: {path}", flush=True)
    return path


def execute(args):
    check_release()
    root = args.output.resolve()
    model_dir, reference_run = args.model_dir.resolve(), args.reference_run.resolve()
    if any(root.is_relative_to(source) or source.is_relative_to(root) for source in (model_dir, reference_run)):
        raise ValueError("Use a separate output directory, outside the model and original result directories")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another controller is active for {root}") from exc
        config_path = root / "run_config.json"
        existing_config = gp.read_json(config_path) if config_path.exists() else None
        if existing_config is not None and existing_config.get("protocol") != gp.PROTOCOL_VERSION:
            raise ValueError(f"Another experiment uses {root}. Choose a separate DialogSum output directory.")
        if existing_config is None:
            previous = [path for path in root.iterdir() if path.name != ".run.lock"]
            status = root / "status.json"
            recoverable = (status.is_file() and gp.read_json(status).get("stage") == "asset_preflight"
                           and all(path == status or path.name.startswith(gp.ARCHIVE_PREFIX)
                                   and path.name.endswith(".tar.gz") for path in previous))
            if previous and not recoverable:
                raise FileExistsError("Use an empty output directory for a new run")
        stage = "asset_preflight"
        try:
            if existing_config is None:
                free = shutil.disk_usage(root).free
                if free < gp.START_DISK:
                    raise OSError(f"Need 64 GiB free at {root} before creating new weights; found {free/gp.GIB:.1f} GiB")
            config = resolve_inputs(args)
            if existing_config is not None:
                if existing_config != config:
                    raise ValueError("Paths, source hashes or package changed. Use a new output directory")
            else:
                gp.write_json(config_path, config)
            if args.command == "confirm":
                pilot = root / "pilot_summary.json"
                if not pilot.is_file() or gp.read_json(pilot).get("status") != "COMPLETED":
                    raise RuntimeError("Complete and review the pilot before running confirmation")
            link_paper_assets(root, config)
            for job in PREPARE + gp.schedule("pilot"):
                stage = job["key"]
                phase(root, job, args, lock.fileno())
            if args.command == "confirm":
                for job in gp.schedule("confirm"):
                    stage = job["key"]
                    phase(root, job, args, lock.fileno())
            stage = "summarize"
            summary = gp.summarize(root, args.command)
            gp.write_json(root / "status.json", dict(status=summary["status"], scope=args.command))
            print("STATUS:", summary["status"], flush=True)
            archive(root, args.command)
            return 0 if summary["status"] == "COMPLETED" else 2
        except (Exception, KeyboardInterrupt) as exc:
            gp.write_json(root / "status.json", dict(status="STOPPED", scope=args.command, stage=stage,
                error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            print(f"STATUS: STOPPED at {stage} | {exc}", flush=True)
            archive(root, args.command)
            return 1


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "_worker":
        from generation_worker import run
        root, attempt = map(Path, sys.argv[2:])
        config = gp.read_json(root / "run_config.json")
        config.update(gp.read_json(attempt / "settings.json"))
        run(root, gp.read_json(attempt / "job.json"), attempt, config)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check-assets", "pilot", "confirm"))
    parser.add_argument("--model-dir", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the three pinned DialogSum JSONL files")
    parser.add_argument("--idle-timeout", type=int, default=8*3600, help="Seconds to wait per phase")
    parser.add_argument("--phase-timeout", type=int, default=8*3600, help="Seconds allowed per worker")
    args = parser.parse_args()
    if min(args.idle_timeout, args.phase_timeout) < 1:
        parser.error("Timeouts must be positive")
    if args.command == "plan":
        for job in PREPARE + gp.schedule("pilot") + gp.schedule("confirm"):
            print(job["key"])
        return 0
    if args.command == "check-assets":
        check_release()
        config = resolve_inputs(args)
        print(f"Verified {len(config['pinned_assets'])} pinned input files.")
        return 0
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
