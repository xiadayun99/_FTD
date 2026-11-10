#!/usr/bin/env python3
"""Export verified assets, then independently admit/replay canonical switching.

The numerical and scoring files in vendor/ are unchanged from the reviewed RTX
release. A finite/integrity preflight can select a different, explicitly logged
GEMM reduction/attention execution contract, uniformly for every method.
It never installs packages, trains adapters, copies RTX corrections, or kills
foreign processes. A failed/unavailable measurement remains in the archive.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
MIB = 2**20
CAP = 2560 * MIB
TASKS = ("mrpc", "qnli", "agnews")
LABELS = {"mrpc": {"equivalent", "not_equivalent"},
          "qnli": {"entailment", "not_entailment"},
          "agnews": {"world", "sports", "business", "science"}}
VERSION = "ftd_crossdevice_numerical_20261104_v3"
SOURCE = Path.home()/"ftd_strong_vllm_outputs/ftd_strong_vllm_20261104T020754Z_e4opuion"
SOURCE_MANIFEST = "afaf1bad0b84c2bace9fcf6b73d2ee654a1a3f84768a27b671d5542014871e67"
METHODS = {"direct": ("canonical_patch_cached", "direct"),
           "node": ("canonical_patch_cached", "node"),
           "native": ("hf_native_cached", "node"),
           "reference": ("host_copy_cached", "node")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+"\n", encoding="utf-8")


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(MIB), b""):
            h.update(data)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def verify_inventory(root, inventory):
    root = Path(root).resolve()
    if not inventory:
        raise ValueError("Empty inventory")
    for rel, expected in inventory.items():
        p = root / rel
        if (Path(rel).is_absolute() or ".." in Path(rel).parts
                or not p.is_file() or p.is_symlink()
                or not p.resolve().is_relative_to(root) or sha(p) != expected):
            raise ValueError("Missing/changed file: " + str(p))


def verify_release():
    verify_inventory(HERE, read(HERE/"RELEASE_MANIFEST.json"))
    verify_inventory(HERE/"vendor", read(HERE/"vendor/PINNED.json"))


def load_core():
    vendor = HERE/"vendor"
    sys.path[:0] = [str(vendor), str(vendor/"engine")]
    core = module("crossdevice_core", vendor/"engine/run_canonical_patch.py")
    import rounding_patch
    compact = module("crossdevice_compact", vendor/"compact_group.py")
    rounding_patch.PatchGroup = compact.CompactPatchGroup
    integration = module("crossdevice_integration", vendor/"request_runtime.py")
    core.TraceExecutor = integration.make_executor_class(core.TraceExecutor, TASKS)
    # Frozen helpers read this value only when constructing a trace.
    # The sealed input is the same 384-question replay, not new test questions.
    core.ROOT = vendor/"engine"
    e, c, p, o = core.helpers()
    return core, e, c, p, o


def schedule():
    return [dict(tag=f"r{b}_{m}", method=m, block=b, kind="timed")
            for b, order in enumerate(itertools.permutations(("direct", "node", "native")))
            for m in order]


def copy_asset(source, destination):
    # Hugging Face snapshots may contain symlinks. Export actual bytes so the
    # portable bundle does not depend on a server cache path.
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_assets(parent, output):
    """CPU/file-only preparation. No model execution or correction export."""
    verify_release()
    parent = parent.resolve()
    if sha(parent/"manifest.json") != SOURCE_MANIFEST:
        raise ValueError("Use the reviewed completed strong-vLLM output: " + str(parent))
    verify_inventory(parent, read(parent/"manifest.json"))
    if read(parent/"status.json")["status"] != "BALANCED_VLLM_COMPARISON_COMPLETED":
        raise ValueError("Parent comparison is incomplete")
    spec = read(parent/"direct_shared/spec.json")
    frozen = read(parent/"method_freeze.json")
    ledger = frozen["method"]["asset_file_sha256"]
    if any(spec[k] != frozen["method"]["assets"][k] for k in ("model_dir", "taskpool_dir", "source_result")):
        raise ValueError("Source asset paths differ from the reviewed frozen method")
    if sha(parent/"engine/input/workload.json") != sha(HERE/"input/workload.json"):
        raise ValueError("Reviewed replay questions differ")
    output.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="crossdevice_export_", dir=output))
    ready = session/"FTD_CrossDevice_Ready_20261104"
    ready.mkdir()
    for path in HERE.rglob("*"):
        rel = path.relative_to(HERE)
        if (path.is_file() and "__pycache__" not in rel.parts
                and rel.parts[0] not in ("manuscript", "tests")):
            copy_asset(path, ready/rel)
    files = {}
    model = Path(spec["model_dir"])
    for name in sorted(ledger):
        path = Path(name)
        if path.parent == model:
            files["assets/model/"+path.name] = path
    if "assets/model/config.json" not in files or not any(k.endswith(".safetensors") for k in files):
        raise ValueError("Missing Qwen3 model config/weights")
    for task in TASKS:
        directory = Path(spec["taskpool_dir"])/"lora"/task
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            files[f"assets/taskpool/lora/{task}/{name}"] = directory/name
    reference = Path(spec["source_result"])
    files["assets/reference/target_metadata.json"] = reference/"target_metadata.json"
    for task in TASKS:
        files[f"assets/reference/reference_slices/{task}.safetensors"] = reference/"reference_slices"/(task+".safetensors")
    total = sum(path.stat().st_size for path in files.values())
    required = 2*total + 1024*MIB
    if shutil.disk_usage(output).free < required:
        raise OSError("Export needs room for asset copies and archive; free at least " + str(required//MIB) + " MiB")
    manifest = {}
    for rel, path in files.items():
        print("PACK:", rel, flush=True)
        copy_asset(path, ready/rel)
        manifest[rel] = sha(path)
        if str(path) in ledger:
            if manifest[rel] != ledger[str(path)]:
                raise ValueError("Asset changed since method freeze: " + str(path))
        elif not rel.startswith("assets/reference/reference_slices/"):
            raise ValueError("Asset lacks the reviewed frozen identity: " + str(path))
    dump(ready/"ASSET_MANIFEST.json", manifest)
    dump(ready/"ASSET_PROVENANCE.json", dict(version=VERSION,
         source_result=str(parent), source_manifest_sha256=SOURCE_MANIFEST,
         asset_bytes=total, original_server_paths=spec,
         source_method_freeze_id=frozen["method_freeze_id"],
         transferred_rtx_correction_payloads=False,
         canonical_target_bits_preserved=True,
         target_device_predictors_and_corrections_must_be_rebuilt=True))
    # The export is itself a new sealed release; the original ZIP is untouched.
    dump(ready/"RELEASE_MANIFEST.json", {str(f.relative_to(ready)): sha(f)
         for f in sorted(ready.rglob("*")) if f.is_file() and f.name != "RELEASE_MANIFEST.json"})
    archive = session/"FTD_CrossDevice_Ready_20261104.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tf:
        tf.add(ready, arcname=ready.name, recursive=True)
    print("PORTABLE_BUNDLE:", archive, flush=True)
    print("Copy this bundle to Jetson; do not send its model weights back to chat.", flush=True)


def runtime_contract(torch, e, p, numerical_profile="original_sdpa"):
    torch.manual_seed(0)
    e.torch_runtime(torch)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    p.cache_runtime_contract()
    import numerical_guard as ng
    numerical_settings = ng.configure(torch,numerical_profile)
    import importlib.metadata as md
    import triton
    driver = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                                text=True, capture_output=True, timeout=15)
        driver = result.stdout.strip() if result.returncode == 0 else None
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                triton=triton.__version__, transformers=md.version("transformers"),
                peft=md.version("peft"), gpu=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)), driver=driver,
                machine=platform.machine(), platform=platform.platform(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                l4t=Path("/etc/nv_tegra_release").read_text().strip() if Path("/etc/nv_tegra_release").exists() else None,
                cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                sdp_flash=torch.backends.cuda.flash_sdp_enabled(),
                sdp_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                sdp_math=torch.backends.cuda.math_sdp_enabled(), deterministic=True, tf32=False,
                numerical_source_hashes=read(HERE/"vendor/PINNED.json"),
                preparation_io_sha256=sha(HERE/"bounded_io.py"),
                numerical_preflight_sha256=sha(HERE/"numerical_guard.py"),
                **numerical_settings)


def observe_gpu(group=None):
    """NVML where supported; scoped /proc observations on Jetson otherwise."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(0)
            processes = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            foreign = []
            for proc in processes:
                try:
                    if group is None or os.getpgid(proc.pid) != group:
                        foreign.append(proc.pid)
                except ProcessLookupError:
                    pass
            return dict(backend="nvml", status="ok", foreign_pids=foreign,
                        process_enumeration_verified=True)
        finally:
            nv.nvmlShutdown()
    except Exception as exc:
        nvml_reason = type(exc).__name__
    foreign = []
    other_clients = []
    unreadable = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if group is not None and os.getpgid(int(proc.name)) == group:
                continue
            fd = proc/"fd"
            targets = [os.readlink(f) for f in fd.iterdir()]
            if any("/dev/nvhost-gpu" in t or t == "/dev/nvidia0" for t in targets):
                # A graphics device handle alone is not a CUDA context. Keep
                # graphics/unknown clients in observations; conservatively
                # block a GPU client that also maps CUDA execution libraries.
                try:
                    maps = (proc/"maps").read_text(errors="replace")
                    uses_cuda = any(x in maps for x in ("libcuda.so", "libcudart.so", "libtorch_cuda.so"))
                except (PermissionError, FileNotFoundError, ProcessLookupError):
                    uses_cuda = False
                if uses_cuda:
                    foreign.append(int(proc.name))
                else:
                    other_clients.append(int(proc.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            unreadable += 1
    return dict(backend="proc_gpu_fd", status="limited", foreign_pids=foreign,
                process_enumeration_verified=False, inaccessible_processes=unreadable,
                graphics_or_unclassified_gpu_clients=other_clients,
                nvml_unavailable_reason=nvml_reason,
                interpretation="Open GPU-device handles only; absence is not complete GPU isolation evidence")


def require_no_observed_foreign(group=None):
    result = observe_gpu(group)
    if result["foreign_pids"]:
        raise RuntimeError("Observed foreign GPU device users: " + str(result["foreign_pids"]) + "; no process was stopped")
    return result


class Telemetry:
    def __init__(self, directory):
        self.directory = directory
        self.stop = threading.Event()
        self.thread = None
        self.proc = None
        self.stream = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if shutil.which("tegrastats"):
            self.stream = (self.directory/"tegrastats.log").open("w")
            self.proc = subprocess.Popen(["tegrastats", "--interval", "1000"],
                                         stdout=self.stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        def sample():
            with (self.directory/"telemetry.jsonl").open("w") as f:
                while not self.stop.is_set():
                    item = dict(monotonic_ns=time.perf_counter_ns(),
                                host_loadavg=list(os.getloadavg()),
                                host_memory=host_memory(),
                                observation=observe_gpu(os.getpgrp()))
                    f.write(json.dumps(item)+"\n");f.flush()
                    self.stop.wait(1.)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.proc is not None:
            self.proc.terminate()  # Only this wrapper's own tegrastats child.
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill();self.proc.wait(timeout=3)
            self.stream.close()


def doctor(root):
    result = dict(version=VERSION, python=sys.version, executable=sys.executable,
                  machine=platform.machine(), issues=[], automatic_installs=False)
    for name in ("torch", "transformers", "peft", "safetensors", "numpy", "triton"):
        try:
            mod = __import__(name)
            result[name] = getattr(mod, "__version__", "imported")
        except Exception as exc:
            result["issues"].append(name + ": " + type(exc).__name__ + ": " + str(exc))
    if sys.version_info < (3, 9):
        result["issues"].append("Python >=3.9 required")
    if not (HERE/"ASSET_MANIFEST.json").exists():
        result["issues"].append("Run export on the RTX server and unpack its PORTABLE_BUNDLE here first")
    else:
        try:
            verify_inventory(HERE, read(HERE/"ASSET_MANIFEST.json"))
        except Exception as exc:
            result["issues"].append(str(exc))
    if not result["issues"]:
        try:
            core, e, c, p, o = load_core()
            import torch
            result["contract"] = runtime_contract(torch, e, p)
            result["gpu_observation"] = require_no_observed_foreign(os.getpgrp())
            from safetensors import safe_open
            from safetensors.torch import save_file
            fixture = torch.tensor([0., -0., 1., -2.], dtype=torch.float16)
            fixture_path = root/"reference_view_fixture.safetensors"
            save_file({"weight":fixture}, str(fixture_path))
            with safe_open(str(fixture_path), framework="pt", device="cpu") as handle:
                view = handle.get_tensor("weight")
            import gc
            del handle;gc.collect()
            if not torch.equal(view.view(torch.int16), fixture.view(torch.int16)):
                raise ValueError("Reference tensor view did not preserve raw bits after reader close")
            result["reference_view_lifetime_raw_bits_checked"] = True
            # Compile/execute both the per-projection and compact grouped
            # decoder on tiny raw-bit fixtures before loading the large model.
            result["cuda_decoder_vectors"] = core.validate_cuda_decoder_vectors(torch)
            torch.cuda.synchronize()
            result["free_cuda_bytes"], result["total_cuda_bytes"] = torch.cuda.mem_get_info()
        except Exception as exc:
            result["issues"].append(type(exc).__name__ + ": " + str(exc))
            result["traceback"] = traceback.format_exc()
    result["status"] = "READY" if not result["issues"] else "BLOCKED_ENVIRONMENT"
    dump(root/"doctor.json", result)
    return result


def make_spec(method):
    policy, operator = METHODS[method]
    return dict(version=VERSION, model_dir=str(HERE/"assets/model"),
                taskpool_dir=str(HERE/"assets/taskpool"), source_result=str(HERE/"assets/reference"),
                cap_bytes=CAP, kv_bank_cap_bytes=CAP, cache_budget_policy="shared",
                canonical_transition_operator=operator, collect_dead_kv_versions=False,
                limit=384, policy=policy, new_independent_test=False,
                reused_previously_observed_inputs=True, project_wide_blind_test=False)


def full_path_checks(torch, e, parameters, targets, make):
    checks = []
    for source in TASKS:
        for target in TASKS:
            if source == target:
                continue
            e.install_host_target(torch, parameters, targets[source]);state = make(source)
            state.apply(target);torch.cuda.synchronize()
            actual = e.tensor_digest(torch, [(n, parameters[n]) for n in targets[target]])
            if actual != e.LOCAL_TARGET_HASHES[target]:
                raise ValueError("Full mutable-weight edge hash differs")
            checks.append(dict(source=source, target=target, mutable_weights_sha256=actual))
            del state
    e.install_host_target(torch, parameters, targets["mrpc"]);state = make("mrpc")
    for target in ("qnli", "agnews", "mrpc")*6:
        state.apply(target);torch.cuda.synchronize()
        actual = e.tensor_digest(torch, [(n, parameters[n]) for n in targets[target]])
        if actual != e.LOCAL_TARGET_HASHES[target]:
            raise ValueError("Continuous grouped path hash differs")
        checks.append(dict(target=target, cycle=True, mutable_weights_sha256=actual))
    return checks


def host_memory():
    import bounded_io
    return bounded_io.memory_snapshot()


def require_host_headroom():
    value = host_memory()
    if value["MemAvailable_bytes"] < 512*MIB:
        raise MemoryError("Available host/unified memory fell below the 512 MiB setup floor: " + str(value))
    return value


def load_reference_views(torch, e):
    """Keep safe_open tensor views, avoiding 2.46 GiB of eager CPU clones.

    This is an asset-loader change for unified-memory devices, not a numerical
    operator or scorer change. All target bits still receive full hashes.
    The views are never modified; installation always writes model parameters.
    Report logical reference bytes, without claiming fixed physical residency.
    """
    directory = HERE/"assets/reference"
    metadata = e.read_signed(directory/"target_metadata.json")
    targets = {}
    for task in TASKS:
        reader = e.TensorFiles(directory/"reference_slices"/(task+".safetensors"))
        with reader.open(list(reader.entries)) as get:
            values = {name:get(name) for name in reader.entries}
        if any(not t.is_contiguous() or t.dtype != torch.float16 for t in values.values()):
            raise ValueError("Reference slice layout/dtype differs")
        actual = e.tensor_digest(torch, values.items())
        if actual != metadata["target_slice_sha256"][task] or actual != e.LOCAL_TARGET_HASHES[task]:
            raise ValueError("Full canonical reference hash differs: " + task)
        targets[task] = values
    amount = sum(t.numel()*t.element_size() for vals in targets.values() for t in vals.values())
    if amount > e.HOST_REFERENCE_BUDGET_BYTES:
        raise MemoryError("Logical host reference budget exceeded")
    return targets, metadata


def worker(root, job):
    import torch
    import gc
    import bounded_io as bio
    import numerical_guard as ng
    core, e, c, p, o = load_core()
    import rounding_patch as rp
    from direct_transition import PairFactors, build_banks, validate_primitives, DirectSwitch
    from compact_group import CompactPatchGroup
    directory = root/"stages"/job["tag"]
    directory.mkdir(parents=True, exist_ok=False)
    with Telemetry(directory):
        isolation = require_no_observed_foreign(os.getpgrp())
        numerical_profile = read(root/"protocol.json")["numerical_profile"]
        contract = runtime_contract(torch, e, p,numerical_profile)
        if contract != read(root/"protocol.json")["contract"]:
            raise ValueError("Worker numerical contract differs from selected preflight")
        spec = make_spec(job["method"])
        setup_memory_before = require_host_headroom()
        io_check = bio.check_roundtrip(torch,core,rp,directory/"io_fixture",contract)
        targets, meta = bio.load_targets(torch,e,HERE/"assets/reference")
        native = job["method"] == "native"
        need = job["method"] in ("direct", "node")
        model, factors, base = core.load_base(spec, torch, e, targets, native=native, need_factors=need)
        effective_attention = ng.configure_model(model,numerical_profile)
        bio.trim_heap()
        setup_memory_loaded = require_host_headroom()
        params = dict(model.named_parameters())
        banks = None
        if job["kind"] == "admission":
            began = time.perf_counter_ns()
            admission = root/"admission"/job["method"]
            admission.mkdir(parents=True,exist_ok=False)
            if job["method"] == "direct":
                pairs = PairFactors(torch,factors)
                del factors;factors = pairs
            projections = len(targets["mrpc"])
            descriptor_bytes = 6*(4*8*projections+3*4*projections+2*(projections+1)*4)
            fixed = e.model_payload_bytes(model)+factors.bytes+descriptor_bytes
            encoder = bio.SpoolingEncoder(torch,rp.encode,admission/"codec_spool",
                admission/"encoding_progress.jsonl",fixed,CAP)
            try:
                if job["method"] == "node":
                    with encoder.install(rp):
                        banks, records, encoding_seconds = core.build_patches(torch,params,factors,base,targets)
                else:
                    banks, records = build_banks(torch,params,factors,targets,encoder)
                    encoding_seconds = (time.perf_counter_ns()-began)/1e9
                bio.serialize(admission,banks,contract)
                dump(admission/"encoding_records.json",records)
                setup = dict(encoding_seconds=encoding_seconds,
                    build_serialize_wall_seconds=(time.perf_counter_ns()-began)/1e9,
                    payload_bytes=encoder.payload,fixed_state_before_corrections_bytes=fixed,
                    spooled_projection_count=encoder.count,
                    note="Includes bounded disk spooling and setup-only heap release; not comparable to RTX setup timing")
                del banks
            except bio.PreparationResourceLimit as exc:
                dump(directory/"report.json",dict(status="INFEASIBLE",reason=str(exc),
                    job=job,runtime_contract=contract,target_hashes=e.LOCAL_TARGET_HASHES,
                    phase="preparation",resource_limit=exc.details,io_roundtrip_check=io_check,
                    retained_state_only_is_not_physical_memory=True))
                return
            finally:
                shutil.rmtree(admission/"codec_spool",ignore_errors=True)
                bio.trim_heap()
            torch.cuda.empty_cache()
        else:
            setup = None
        validation = None;checks = []
        if need:
            banks, patch_meta = bio.deserialize(root/"admission"/job["method"],torch,rp.Patch,contract)
            if job["method"] == "direct":
                if job["kind"] != "admission":
                    pairs = PairFactors(torch, factors)
                    del factors;factors = pairs
                validation = validate_primitives(torch, params, factors, targets, banks, rp.bit_equal, repeats=1)
                make = lambda initial: DirectSwitch(torch, params, factors, banks, initial, contract, CompactPatchGroup)
            else:
                validation = core.validate_primitives(torch, params, factors, base, targets, banks, repeats=1)
                make = lambda initial: rp.CanonicalSwitch(params, factors, banks["unmerge"], banks["merge"], initial, contract)
            checks = full_path_checks(torch, e, params, targets, make)
        dump(directory/"allocation_validation.json",dict(
            allocation_primitive_validation=validation,full_grouped_path_checks=checks,
            target_hashes=e.LOCAL_TARGET_HASHES))
        common_check = None if native else ng.shared_state(torch,e,model,targets,meta,
            directory/"shared_state_check.json")
        # Persist this BEFORE any scoring, including failures in warmup.
        dump(directory/"pre_trace_validation.json",dict(
            job=job,runtime_contract=contract,effective_attention_implementation=effective_attention,
            allocation_primitive_validation=validation,full_grouped_path_checks=checks,
            shared_state_check=common_check,io_roundtrip_check=io_check,
            setup=setup,target_hashes=e.LOCAL_TARGET_HASHES))
        if job["kind"] == "admission":
            ng.warmup_check(sys.modules[__name__],torch,core,e,c,p,o,spec,model,
                            factors,targets,base,contract,directory,banks)
            bio.trim_heap()
        setup_memory_before_trace = require_host_headroom()
        # Use the pinned entry point; only its sealed workload path is supplied.
        # Keep a local input copy beside the frozen core; release pins cover it.
        try:
            report = core.execute_trace(directory, spec, METHODS[job["method"]][0], model, factors,
                        targets, base, contract, e, c, p, o, banks=banks,
                        limit=32 if job["kind"] == "admission" else 384,
                        write_tag="smoke" if job["kind"] == "admission" else "trace")
        except (MemoryError, torch.OutOfMemoryError) as exc:
            report = dict(status="INFEASIBLE", reason=str(exc), logical_cap_bytes=CAP)
        except Exception as exc:
            dump(directory/"trace_failure.json",dict(error_type=type(exc).__name__,reason=str(exc),
                traceback=traceback.format_exc(),pre_trace_validation_saved=True))
            raise
        report.update(job=job, runtime_contract=contract,
                      allocation_primitive_validation=validation, full_grouped_path_checks=checks,
                      target_hashes=e.LOCAL_TARGET_HASHES, setup=setup, observation=isolation,
                      setup_memory=dict(before=setup_memory_before, loaded=setup_memory_loaded,
                                        before_trace=setup_memory_before_trace),
                      io_roundtrip_check=io_check,
                      shared_state_check=common_check,
                      effective_attention_implementation=effective_attention,
                      reference_loader="explicit numpy.memmap CPU views; full target hash checks",
                      cross_device_reference_bits=True, rtx_payload_reused=False)
        dump(directory/"report.json", report)


def audit_records(rr, cases, report, method):
    if len(rr) != len(cases) or report["requests"] != len(cases):
        raise ValueError("Incomplete request records")
    previous_end = None
    for row, case in zip(rr, cases):
        if any(row[k] != case[k] for k in ("position", "request_key", "case_id", "task", "gold")):
            raise ValueError("Question identities differ")
        elapsed = (row["finished_monotonic_ns"]-row["started_monotonic_ns"])/1e6
        if (elapsed < 0 or not math.isclose(elapsed, row["request_ms"], abs_tol=1e-8)
                or previous_end is not None and row["started_monotonic_ns"] < previous_end):
            raise ValueError("Timing boundaries differ or overlap")
        previous_end = row["finished_monotonic_ns"]
        parts = ("tokenize_ms", "parameter_ms", "decision_ms", "score_ms", "admission_ms")
        if (any(not math.isfinite(row[k]) or row[k] < 0 for k in parts)
                or not math.isclose(sum(row[k] for k in parts), row["request_ms"], abs_tol=1e-8)):
            raise ValueError("Timing component accounting differs")
        if (set(row["scores"]) != LABELS[row["task"]]
                or any(not math.isfinite(v) for v in row["scores"].values())
                or row["prediction"] not in LABELS[row["task"]]):
            raise ValueError("Candidate scores or label differ")
        winner = max(sorted(row["scores"]), key=lambda k: row["scores"][k])
        if row["prediction"] != winner:
            raise ValueError("Prediction does not follow the frozen score/tie rule")
        if row["approximate_cache_used"] or (row["cache_hit"] and row["cache_origin_version"] != row["state_after"]):
            raise ValueError("Unexpected producer identity mismatch")
        if row["logical_live_max_bytes"] > CAP or report["logical_live_max_bytes"] > CAP:
            raise ValueError("Logical tensor budget exceeded")
        if method in ("direct", "node", "reference"):
            if row["state_after"] != "canonical:"+report["target_hashes"][row["task"]]:
                raise ValueError("Canonical state label differs")
        if method in ("direct", "node") and (row["host_target_copy_bytes"] or row["device_dense_copy_bytes"]):
            raise ValueError("Dense state copy inside canonical timed trace")
    if method in ("direct", "node"):
        checks = report["full_grouped_path_checks"]
        edges = {(x.get("source"), x["target"]) for x in checks if "source" in x}
        if (report["allocation_primitive_validation"].get("all_equal") is not True
                or len(checks) != 24 or edges != {(s,t) for s in TASKS for t in TASKS if s != t}
                or [x["target"] for x in checks if x.get("cycle")] != ["qnli", "agnews", "mrpc"]*6
                or any(x["mutable_weights_sha256"] != report["target_hashes"][x["target"]] for x in checks)):
            raise ValueError("Full-projection/grouped-path admission missing")
        calls = (196 if method == "direct" else 392)*report["switches"]
        if rr[-1]["issued_addmm_calls"] != calls:
            raise ValueError("Issued API update counts differ")
    if method != "native" and report["final_mutable_weights_sha256"] != report["target_hashes"][report["final_task"]]:
        raise ValueError("Final full mutable-weight hash differs")
    return True


def audit_stage(root, job):
    directory = root/"stages"/job["tag"]
    report = read(directory/"report.json")
    if report["job"] != job:
        raise ValueError("Stage identity differs")
    protocol = read(root/"protocol.json")
    if report["runtime_contract"] != protocol["contract"]:
        raise ValueError("Device/software arithmetic contract changed")
    targets = read(HERE/"assets/reference/target_metadata.json")["target_slice_sha256"]
    if report["target_hashes"] != targets:
        raise ValueError("Admitted target hashes differ from the sealed asset reference")
    if report["status"] == "INFEASIBLE":
        return dict(job=job, status="INFEASIBLE", reason=report["reason"]), None
    if report["status"] != "COMPLETED":
        raise ValueError("Unexpected stage status")
    admission = job["kind"] == "admission"
    rr = rows(directory/("smoke_requests.jsonl" if admission else "trace_requests.jsonl"))
    work = read(HERE/"input/workload.json")
    audit_records(rr, work["check"][:32] if admission else work["check"], report, job["method"])
    if admission:
        checks = report["smoke_full_weight_audit"]
        if len(checks) != 32 or any(x["position"] != case["position"] or x["mutable_weights_sha256"] != report["target_hashes"][case["task"]]
                                    for x, case in zip(checks, work["check"][:32])):
            raise ValueError("Audited 32-request smoke incomplete")
    samples = rows(directory/"telemetry.jsonl")
    if not samples or any(r["observation"]["foreign_pids"] for r in samples):
        raise ValueError("Missing telemetry or observed foreign GPU device user")
    values = sorted(x["request_ms"] for x in rr)
    at = (len(values)-1)*.95;lo = math.floor(at);hi = math.ceil(at)
    p95 = values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(job=job, status="COMPLETED", requests=len(rr),
                mean_ms=statistics.mean(values), p95_ms=p95,
                correct=sum(r["prediction"] == r["gold"] for r in rr),
                parameter_mean_ms=statistics.mean(r["parameter_ms"] for r in rr),
                fixed_tensor_bytes=report["fixed_tensor_bytes"],
                logical_live_max_bytes=report["logical_live_max_bytes"],
                allocator_peak_bytes=report["allocator_peak_bytes"],
                reserved_peak_bytes=report["reserved_peak_bytes"],
                telemetry_samples=len(samples), isolation_enumeration_verified=all(
                    x["observation"]["process_enumeration_verified"] for x in samples)), rr


def run_stage(root, job, timeout):
    directory = root/"stages"/job["tag"]
    if (directory/"report.json").exists():
        audit_stage(root, job)
        sessions = read(root/"stage_sessions.json")
        if job["tag"] not in sessions:
            sessions[job["tag"]] = "recovered_unknown_"+job["tag"]
            dump(root/"stage_sessions.json", sessions)
        return
    if directory.exists():
        old = root/"interrupted"/(job["tag"]+"_"+uuid.uuid4().hex[:8])
        old.parent.mkdir(parents=True, exist_ok=True);shutil.move(directory, old)
    if job["kind"] == "admission" and (root/"admission"/job["method"]).exists():
        old = root/"interrupted"/(job["method"]+"_payload_"+uuid.uuid4().hex[:8])
        old.parent.mkdir(parents=True, exist_ok=True);shutil.move(root/"admission"/job["method"], old)
    require_no_observed_foreign()
    print("STAGE:", job["tag"], flush=True)
    dump(root/"active_stage.json", job)
    log = root/"logs"/(job["tag"]+"_"+uuid.uuid4().hex[:6]+".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    launched_ns = time.perf_counter_ns()
    with log.open("w") as f:
        process = subprocess.Popen([sys.executable, str(HERE/"run_crossdevice.py"), "_worker",
                                    "--result-dir", str(root)], stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Kill only this invocation's owned worker group and its telemetry.
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL);process.wait(timeout=5)
            raise
    import signal
    dump(root/"process_exits"/(job["tag"]+"_"+uuid.uuid4().hex[:8]+".json"),dict(
        job=job,pid=process.pid,returncode=process.returncode,
        signal=signal.Signals(-process.returncode).name if process.returncode<0 else None,
        launched_monotonic_ns=launched_ns,finished_monotonic_ns=time.perf_counter_ns(),log=str(log)))
    if process.returncode:
        raise RuntimeError("Worker exit code="+str(process.returncode)+"; "+str(log)+"\n"+"\n".join(log.read_text(errors="replace").splitlines()[-18:]))
    audit_stage(root, job)
    sessions = read(root/"stage_sessions.json")
    sessions[job["tag"]] = read(root/"active_session.json")
    dump(root/"stage_sessions.json", sessions)


def summarize(root):
    jobs = schedule()
    metrics = [];raw = {}
    protocol = read(root/"protocol.json")
    reference = rows(root/protocol["quality_reference_rows"])
    previous_end = None
    for job in jobs:
        m, rr = audit_stage(root, job)
        if rr is not None:
            if previous_end is not None and rr[0]["started_monotonic_ns"] < previous_end:
                raise ValueError("Formal stages overlap or differ from registered execution order")
            previous_end = rr[-1]["finished_monotonic_ns"]
            m.update(prediction_changes_vs_device_reference=sum(x["prediction"] != y["prediction"] for x,y in zip(rr, reference)),
                     true_label_gains=sum(y["prediction"] != y["gold"] and x["prediction"] == x["gold"] for x,y in zip(rr, reference)),
                     true_label_losses=sum(y["prediction"] == y["gold"] and x["prediction"] != x["gold"] for x,y in zip(rr, reference)),
                     max_score_difference_vs_device_reference=max(abs(x["scores"][k]-y["scores"][k]) for x,y in zip(rr, reference) for k in x["scores"]),
                     per_task_correct={task:sum(x["prediction"] == x["gold"] for x in rr if x["task"] == task) for task in TASKS})
            raw[job["tag"]] = rr
        metrics.append(m)
    aggregates = []
    for method in ("direct", "node", "native"):
        mm = [x for x in metrics if x["job"]["method"] == method]
        a = dict(method=method, statuses=[x["status"] for x in mm])
        if all(x["status"] == "COMPLETED" for x in mm):
            a.update(mean_ms=statistics.mean(x["mean_ms"] for x in mm),
                     mean_block_p95_ms=statistics.mean(x["p95_ms"] for x in mm),
                     correct_each_block=[x["correct"] for x in mm],
                     fixed_tensor_bytes=max(x["fixed_tensor_bytes"] for x in mm),
                     max_allocator_peak_bytes=max(x["allocator_peak_bytes"] for x in mm))
        aggregates.append(a)
    deltas = []
    for b in range(6):
        mm = {x["job"]["method"]:x for x in metrics if x["job"]["block"] == b}
        deltas.append(dict(block=b, node_minus_direct_ms=(mm["node"]["mean_ms"]-mm["direct"]["mean_ms"])
                           if all(mm[m]["status"] == "COMPLETED" for m in ("node", "direct")) else None))
    result = dict(version=VERSION, aggregates=aggregates, stages=metrics, paired_block_deltas=deltas,
                  unique_questions=384, new_questions=0, repeated_inputs_not_independent_accuracy_samples=True,
                  logical_cap_bytes=CAP, physical_memory_limit_claim=False,
                  checked_rtx_bits_on_new_device=True, same_scorer_and_decoder_sources=True,
                  numerical_profile=protocol["numerical_profile"],
                  runtime_contract=protocol["contract"],
                  numerical_profile_selection_used_accuracy_or_latency=False,
                  pool_timings_with_rtx=False,
                  isolation_enumeration_verified=all(x.get("isolation_enumeration_verified", False) for x in metrics),
                  single_invocation=len(set(read(root/"stage_sessions.json").values())) == 1,
                  clock_thermal_observations="See each stage's unfiltered tegrastats.log when available",
                  submission_ready=False)
    dump(root/"summary.json", result)
    return result


def archive_result(root):
    included = [f for f in root.rglob("*") if f.is_file() and f.suffix not in (".safetensors", ".partial", ".bin") and f.name != "manifest.json" and "codec_spool" not in f.parts and "serialization_streams" not in f.parts]
    dump(root/"manifest.json", {str(f.relative_to(root)):sha(f) for f in sorted(included)})
    archive = root.parent/(root.name+"_"+uuid.uuid4().hex[:8]+".tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        for f in [*included, root/"manifest.json"]:
            tf.add(f, arcname=root.name+"/"+str(f.relative_to(root)), recursive=False)
    print("SEND_THIS_ARCHIVE:", archive, flush=True)
    print("RESUME_RESULT_DIR:", root, flush=True)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "doctor", "run", "resume", "_worker", "_probe", "inspect"))
    parser.add_argument("--parent", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path.home()/"ftd_crossdevice_outputs")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--stage-timeout", type=int, default=7200)
    parser.add_argument("--numerical-profile",choices=("original_sdpa","strict_sdpa","strict_eager"))
    parser.add_argument("--probe-method",choices=("reference","native"))
    args = parser.parse_args(argv)
    verify_release()
    if args.mode == "export":
        export_assets(args.parent, args.output_dir)
        return 0
    if args.mode == "_worker":
        worker(args.result_dir.resolve(), read(args.result_dir/"active_stage.json"))
        return 0
    if args.mode == "_probe":
        if args.result_dir is None or args.numerical_profile is None or args.probe_method is None:
            raise ValueError("Internal probe requires its result/profile/method arguments")
        import numerical_guard as ng
        return ng.probe_worker(sys.modules[__name__],args.result_dir.resolve(),
                               args.numerical_profile,args.probe_method)
    if args.mode == "inspect":
        text = read(args.result_dir/"summary.json")
        print(json.dumps(text, ensure_ascii=False, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "resume":
        if args.result_dir is None:
            raise ValueError("resume requires --result-dir")
        root = args.result_dir.resolve()
        protocol = read(root/"protocol.json")
        if protocol["release_manifest_sha256"] != sha(HERE/"RELEASE_MANIFEST.json"):
            raise ValueError("Resume requires unchanged code, workload and assets")
    else:
        root = Path(tempfile.mkdtemp(prefix="ftd_crossdevice_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_"),
                                     dir=args.output_dir))
    print("RESULT_DIR:", root, flush=True)
    if args.mode != "resume":
        for source, name in [(HERE/"RELEASE_MANIFEST.json", "source_release_manifest.json"),
                             (HERE/"vendor/PINNED.json", "numerical_source_pins.json"),
                             (HERE/"ASSET_MANIFEST.json", "asset_manifest.json"),
                             (HERE/"ASSET_PROVENANCE.json", "asset_provenance.json"),
                             (HERE/"assets/reference/target_metadata.json", "reference_metadata.json")]:
            if source.exists():
                shutil.copy2(source, root/name)
    success = False
    try:
        # Doctor is a child too: its CUDA context must not pollute measured
        # memory or trigger the subsequent worker's foreign-process guard.
        if args.mode == "doctor":
            result = doctor(root)
            dump(root/"status.json", dict(status=result["status"]))
            return 0 if result["status"] == "READY" else 2
        if args.mode != "resume":
            check = subprocess.run([sys.executable, str(HERE/"run_crossdevice.py"), "doctor", "--output-dir", str(root/"environment_check")],
                                    capture_output=True, text=True, timeout=min(args.stage_timeout, 600))
            dump(root/"doctor_process.json", dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
            candidates = list((root/"environment_check").glob("*/doctor.json"))
            if check.returncode or len(candidates) != 1:
                raise RuntimeError("Environment check blocked; details are retained in the result archive. No packages were installed.")
            result = read(candidates[0])
            dump(root/"protocol.json", dict(version=VERSION, contract=result["contract"], schedule=schedule(),
                 logical_cap_bytes=CAP, release_manifest_sha256=sha(HERE/"RELEASE_MANIFEST.json"),
                 workload_sha256=sha(HERE/"input/workload.json"),
                 primary_endpoint="Paired block mean request difference Direct vs Node",
                 no_threshold_tuning=True, no_rtx_patch_payload_reuse=True,
                 preparation_changes_only=False,preparation_io_sha256=sha(HERE/"bounded_io.py"),
                 prior_blocked_evidence=["ftd_crossdevice_20261104T074237Z_harckygj",
                                         "ftd_crossdevice_20261104T094517Z_smo8wg64"],
                 scorer_codec_timing_boundaries_and_gpu_budget_unchanged=True,
                 gemm_reduction_and_attention_backend_may_change=True,
                 numerical_profile_not_chosen_by_accuracy_or_latency=True,
                 note="Conditional cross-device admission; preserved RTX canonical target bits",
                 new_independent_test=False, new_questions=0, submission_ready=False))
            dump(root/"stage_sessions.json", {})
            shutil.copy2(HERE/"input/workload.json", root/"workload.json")
            # Preserve the exact executed sources, excluding model and target bytes.
            for f in (HERE/"vendor").rglob("*.py"):
                dest = root/"source"/f.relative_to(HERE)
                dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(f, dest)
            shutil.copy2(HERE/"run_crossdevice.py", root/"source/run_crossdevice.py")
            shutil.copy2(HERE/"bounded_io.py",root/"source/bounded_io.py")
            shutil.copy2(HERE/"numerical_guard.py",root/"source/numerical_guard.py")
        dump(root/"active_session.json", uuid.uuid4().hex)
        import numerical_guard as ng
        selection = ng.select_profile(sys.modules[__name__],root,args.stage_timeout)
        protocol = read(root/"protocol.json")
        protocol.update(numerical_profile=selection["profile"],contract=selection["runtime_contract"],
                        quality_reference_rows=selection["quality_reference_rows"],
                        native_preflight_rows=selection["native_preflight_rows"],
                        numerical_profile_order=list(ng.PROFILE_ORDER))
        dump(root/"protocol.json",protocol)
        admission_limits = []
        for method in ("node", "direct"):
            job = dict(tag="admit_"+method, method=method, block=-1, kind="admission")
            run_stage(root, job, args.stage_timeout)
            m, _ = audit_stage(root, job)
            if m["status"] != "COMPLETED":
                admission_limits.append(dict(method=method,metric=m))
        if admission_limits:
            dump(root/"admission_resource_limits.json",admission_limits)
            raise RuntimeError("Admission resource limit recorded; budget and input set were kept fixed. See admission_resource_limits.json")
        for job in schedule():
            run_stage(root, job, args.stage_timeout)
        result = summarize(root)
        status = "CROSSDEVICE_COMPARISON_COMPLETED" if all(all(s == "COMPLETED" for s in a["statuses"]) for a in result["aggregates"]) else "COMPLETED_WITH_INFEASIBLE_METHOD"
        dump(root/"status.json", dict(status=status))
        success = True
        print("STATUS:", status, flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        dump(root/"status.json", dict(status="BLOCKED_OR_FAILED", error_type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc()))
        print("Stopped:", str(exc), flush=True)
    finally:
        archive_result(root)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
