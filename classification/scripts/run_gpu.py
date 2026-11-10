#!/usr/bin/env python3
"""Rebuild locally admitted banks and replay a paper protocol with exact input assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from paper_protocol import (
    ROOT, SWITCHES, confirmation_jobs, frequency_jobs, reorder_workload, switch_count,
)
from reproduce import read, records, stage_metrics, verify_manifest

GIB = 2**30
ASSET_SPEC = ROOT / "data/paper_assets.json"
BOOT = Path("/proc/sys/kernel/random/boot_id")
VENDOR = ROOT / "runtime/vendor/runtime"
REFERENCE_JOB = dict(key="preflight_reference", phase="trace", tag="preflight_reference", method="host_copy")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def verify_assets(args):
    spec = read(ASSET_SPEC)
    model = args.model_dir.resolve()
    source = args.reference_run.resolve()
    workload = args.workload.resolve()
    files = {model / name: digest for name, digest in spec["model_files_sha256"].items()}
    files.update({source / name: digest for name, digest in spec["reference_run_files_sha256"].items()})
    files[workload] = spec["workload_sha256"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError("Missing paper assets:\n  " + "\n  ".join(missing))
    print(f"Checking SHA-256 for {len(files)} paper assets", flush=True)
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError(f"Paper asset digest differs: {path}")
    work = read(workload)
    cases = work["check"]
    if len(cases) != 384 or len({case["request_key"] for case in cases}) != 384 or any(
        case.get("via") or case["position"] != i for i, case in enumerate(cases)
    ) or switch_count(case["task"] for case in cases) != 136:
        raise ValueError("The workload does not match the measured confirmation cohort")
    return model, source, workload, work


def plan(experiment):
    methods = ("direct", "node", "shared") if experiment == "confirmation" else ("direct", "shared")
    return (
        [dict(key="runtime_probe", phase="probe")]
        + [dict(key=f"build_{method}", phase="build", method=method) for method in methods]
        + [REFERENCE_JOB]
        + (confirmation_jobs() if experiment == "confirmation" else frequency_jobs())
    )


def evaluation(root, job):
    return root / "evaluation" / job.get("regime", "confirmation")


def phase_name(job):
    return f'{job["regime"]}_{job["key"]}' if "regime" in job else job["key"]


def phase_output(root, job):
    source = root / "prepared"
    if job["phase"] == "probe":
        return source / "runtime_contract.json"
    if job["phase"] == "build":
        bank = "shared_target" if job["method"] == "shared" else f'patches/{job["method"]}'
        return source / "artifacts" / bank
    return evaluation(root, job) / "stages" / job["tag"]


def output_inventory(root, path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError(f"Phase has invalid output: {path}")
    return {str(p.relative_to(root)): sha(p) for p in files}


def wait_for_idle(timeout):
    import pynvml as nv
    import bounded_io
    nv.nvmlInit()
    try:
        if nv.nvmlDeviceGetCount() != 1:
            raise ValueError("Expose exactly one NVIDIA GPU to both CUDA and NVML")
        handle = nv.nvmlDeviceGetHandleByIndex(0)
        deadline = time.monotonic() + timeout
        ready_since = None
        last_update = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            gpu_free = nv.nvmlDeviceGetMemoryInfo(handle).free
            host_free = bounded_io.memory_snapshot()["MemAvailable_bytes"]
            busy = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            if not busy and gpu_free >= 48 * GIB and host_free >= 104 * GIB:
                ready_since = now if ready_since is None else ready_since
                if now - ready_since >= 30:
                    return
            else:
                ready_since = None
            if now - last_update >= 60:
                print(f"Waiting for 30 s of GPU readiness: compute processes={len(busy)}, "
                      f"GPU free={gpu_free/GIB:.1f} GiB, host available={host_free/GIB:.1f} GiB",
                      flush=True)
                last_update = now
            time.sleep(2)
    finally:
        nv.nvmlShutdown()
    raise TimeoutError("GPU readiness wait expired. Retry with --resume after resources are available")


def run_phase(root, job, args):
    name = phase_name(job)
    marker = root / "completed" / f"{name}.json"
    output = phase_output(root, job)
    if marker.exists():
        if read(marker) != dict(job=job, outputs=output_inventory(root, output)):
            raise ValueError(f"Completed phase changed: {name}")
        print(f"Verified existing phase: {name}", flush=True)
        return
    if output.exists():
        preserved = root / "incomplete" / f"{name}_{uuid.uuid4().hex[:8]}"
        preserved.parent.mkdir(parents=True, exist_ok=True)
        output.rename(preserved)
    wait_for_idle(args.idle_timeout)
    job_path = root / "jobs" / f"{name}.json"
    write_json(job_path, job)
    log = root / "logs" / f"{name}_{uuid.uuid4().hex[:8]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONFAULTHANDLER="1")
    print(f"Phase {name}. Log {log}", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", str(root), str(job_path)]
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=args.phase_timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"Worker failed in {name} with exit {code}. See {log}")
    write_json(marker, dict(job=job, outputs=output_inventory(root, output)))


def worker(root, job_path):
    from run_8b_disk import port_module
    import disk_engine
    import disk_protocol

    config = read(root / "run_config.json")
    if BOOT.read_text().strip() != config["boot_id"]:
        raise ValueError("The host rebooted. Start a new run to rebuild and admit corrections")
    disk_protocol.MODEL_DIR = Path(config["model_dir"])
    source = root / "prepared"
    job = read(job_path)
    if job["phase"] == "probe":
        torch, _, _, _, _, _, contract = disk_engine.runtime(port_module(), source, initial=True)
        if torch.cuda.device_count() != 1:
            raise ValueError("Expected one CUDA device")
        write_json(source / "runtime_contract.json", contract)
    elif job["phase"] == "build":
        if job["method"] == "shared":
            from standalone_build import build
            build(source, source)
        else:
            disk_engine.Session(source, port_module(), job["method"], with_banks=False).build()
    elif job["phase"] == "trace":
        from confirmation_engine import trace
        # The normal Host-copy control supplies a full local score reference.
        # Admission of the canonical operators occurs outside trace timing.
        trace(evaluation(root, job), source, source, job)
    else:
        raise ValueError(f"Unknown worker phase: {job}")


def summarize(root, jobs):
    import disk_protocol

    def stage(job):
        directory = phase_output(root, job)
        return disk_protocol.rows(directory / "trace_requests.jsonl"), read(directory / "report.json")

    ref, _ = stage(REFERENCE_JOB)
    ref_by_key = {row["request_key"]: row for row in ref}
    measured_reference = list(records(ROOT / "data/reference.jsonl.gz"))
    reference_score_difference = max(
        abs(current["scores"][label] - measured["scores"][label])
        for current, measured in zip(ref, measured_reference)
        for label in current["scores"]
    )
    stages = []
    for job in jobs:
        if "block" not in job:
            continue
        rows, report = stage(job)
        max_diff = 0.0
        pred_changes = 0
        for row in rows:
            ref_row = ref_by_key[row["request_key"]]
            max_diff = max(max_diff, *(
                abs(row["scores"][label] - ref_row["scores"][label]) for label in row["scores"]
            ))
            pred_changes += row["prediction"] != ref_row["prediction"]
        stages.append(dict(
            job=job, max_score_difference_vs_local_host_copy=max_diff,
            prediction_changes_vs_local_host_copy=pred_changes,
            **stage_metrics(rows, report),
        ))
    equal = all(item["max_score_difference_vs_local_host_copy"] == 0
                for item in stages if item["job"]["method"] != "native")
    summary = dict(
        status="COMPLETED" if equal else "CANONICAL_SCORE_DIFFERENCE",
        scope="New GPU execution with locally rebuilt and admitted corrections",
        runtime_contract=read(root / "prepared/runtime_contract.json"),
        formal_stages=len(stages), formal_requests=384 * len(stages),
        reference_max_score_difference_vs_archived_host_copy=reference_score_difference,
        canonical_candidate_scores_equal_to_local_reference=equal, stages=stages,
    )
    write_json(root / "summary.json", summary)
    return summary["status"]


def initialize_run(root, model, source, workload, work, args):
    config = dict(
        model_dir=str(model), reference_run=str(source), workload=str(workload),
        experiment=args.experiment, boot_id=BOOT.read_text().strip(),
        release_sha256=sha(ROOT / "MANIFEST_SHA256.json"),
    )
    manifest = root / "run_config.json"
    if args.resume:
        if not manifest.exists() or read(manifest) != config:
            raise ValueError("Resume requires the same paths, protocol, release and OS boot")
    elif manifest.exists():
        raise FileExistsError("The output directory already contains a run")
    elif shutil.disk_usage(root).free < 64 * GIB:
        raise OSError("Need 64 GiB free in the new run directory")
    prepared = root / "prepared"
    for link, target in (
        (prepared / "trained", source / "trained"),
        (prepared / "artifacts/references", source / "artifacts/references"),
    ):
        link.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise ValueError("Input reference link changed")
        else:
            link.symlink_to(target, target_is_directory=True)
    evaluations = {"confirmation": work}
    if args.experiment == "frequency":
        evaluations.update({regime: reorder_workload(work, regime) for regime in SWITCHES})
    for regime, value in evaluations.items():
        path = root / "evaluation" / regime / "engine/input/workload.json"
        if args.resume:
            if read(path) != value:
                raise ValueError(f"Workload changed for {regime}")
        else:
            write_json(path, value)
    if not args.resume:
        write_json(manifest, config)


def run(args):
    import fcntl

    model, source, workload, work = verify_assets(args)
    root = args.output.resolve()
    if root.is_relative_to(model) or root.is_relative_to(source):
        raise ValueError("Use an output directory separate from the input model and reference assets")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize_run(root, model, source, workload, work, args)
        jobs = plan(args.experiment)
        write_json(root / "status.json", dict(status="RUNNING", experiment=args.experiment))
        try:
            for job in jobs:
                run_phase(root, job, args)
            status = summarize(root, jobs)
            write_json(root / "status.json", dict(status=status, experiment=args.experiment))
        except BaseException as error:
            write_json(root / "status.json", dict(status="INCOMPLETE", reason=str(error)))
            raise
    print(f"{status}. Results in {root}")
    return 0 if status == "COMPLETED" else 2


def main():
    sys.path.insert(0, str(VENDOR))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan", help="print a protocol without loading weights")
    planning.add_argument("--experiment", choices=("confirmation", "frequency"), required=True)
    assets = sub.add_parser("check-assets", help="verify local paper assets without initializing CUDA")
    execution = sub.add_parser("run", help="rebuild banks and run one GPU experiment")
    for entry in (assets, execution):
        entry.add_argument("--model-dir", type=Path, required=True)
        entry.add_argument("--reference-run", type=Path, required=True)
        entry.add_argument("--workload", type=Path, required=True)
    execution.add_argument("--experiment", choices=("confirmation", "frequency"), required=True)
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--resume", action="store_true")
    execution.add_argument("--idle-timeout", type=int, default=3600)
    execution.add_argument("--phase-timeout", type=int, default=8 * 3600)
    internal = sub.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("root", type=Path)
    internal.add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.root, args.job)
        return 0
    if args.command == "plan":
        print(json.dumps(plan(args.experiment), indent=2))
        return 0
    try:
        verify_manifest(ROOT, ROOT / "MANIFEST_SHA256.json")
        if args.command == "check-assets":
            verify_assets(args)
            print("PAPER_ASSETS_VERIFIED")
            return 0
        if min(args.phase_timeout, args.idle_timeout) <= 30:
            raise ValueError("Timeouts must exceed the 30 s idle window")
        return run(args)
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

