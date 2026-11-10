"""Untimed numerical preflight; the frozen scorer and GPU operators are intact.

Profiles are tried in a published order based solely on finite execution and
existing integrity checks. Gold labels, accuracy and elapsed time never choose
a profile. Every formal method then uses exactly the selected profile.
"""
import contextlib
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


PROFILES = {
    "original_sdpa": dict(attention="sdpa", reduced=True, flash=True, efficient=True),
    "strict_sdpa": dict(attention="sdpa", reduced=False, flash=False, efficient=False),
    "strict_eager": dict(attention="eager", reduced=False, flash=False, efficient=False),
}
PROFILE_ORDER = tuple(PROFILES)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def read(path):
    return json.loads(Path(path).read_text())


def configure(torch, name):
    profile = PROFILES[name]
    backend = torch.backends.cuda
    backend.matmul.allow_fp16_reduced_precision_reduction = profile["reduced"]
    backend.enable_flash_sdp(profile["flash"])
    backend.enable_mem_efficient_sdp(profile["efficient"])
    backend.enable_math_sdp(True)
    # This optional switch is absent from some NVIDIA 2.5 alpha builds.
    # When available, explicitly disallow reduced-precision math SDPA.
    setter = getattr(backend, "allow_fp16_bf16_reduction_math_sdp", None)
    getter = getattr(backend, "fp16_bf16_reduction_math_sdp_allowed", None)
    if name != "original_sdpa" and setter is not None:
        setter(False)
    return dict(numerical_profile=name, attention_implementation=profile["attention"],
                math_sdp_reduced_precision_control_available=setter is not None,
                math_sdp_reduced_precision=getter() if getter is not None else None)


def configure_model(model, name):
    expected = PROFILES[name]["attention"]
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    current = getattr(backbone.config, "_attn_implementation", None)
    if current != expected:
        setter = getattr(backbone, "set_attn_implementation", None)
        if setter is None:
            raise RuntimeError("Model lacks the official attention implementation setter")
        setter(expected)
    actual = getattr(backbone.config, "_attn_implementation", None)
    if actual != expected:
        raise ValueError("Effective attention backend differs from registered profile")
    return actual


def shared_state(torch, e, model, targets, metadata, result_path=None):
    mutable = set(targets["mrpc"])
    actual = e.tensor_digest(torch,
        [(n, t) for n, t in model.named_parameters() if n not in mutable]
        + [("buffer:"+n, t) for n, t in model.named_buffers()])
    result = dict(actual_sha256=actual, expected_sha256=metadata["shared_state_sha256"],
                  equal=actual == metadata["shared_state_sha256"])
    if result_path is not None:
        dump(result_path,result)
    if not result["equal"]:
        raise ValueError("Non-mutable model tensors changed after GPU path checks: "+str(result))
    return result


def case_identity(case, phase):
    return {"phase": phase, **{k: case[k] for k in
        ("position", "case_id", "request_key", "task", "head_id")}}


def tensors_in(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors_in(child, torch)
    elif isinstance(value, dict):
        for child in value.values():
            # Never traverse a KV cache object or retain its storage.
            yield from tensors_in(child, torch)


class FiniteTracker:
    """Forward output hooks used ONLY while replaying an already failed call."""
    def __init__(self, torch, model):
        self.torch, self.model = torch, model
        self.handles = []
        self.first = None
        self.checked_tensors = 0

    def __enter__(self):
        def hook(name):
            def check(module, args, output):
                if self.first is not None:
                    return
                for tensor in tensors_in(output, self.torch):
                    if not tensor.is_floating_point():
                        continue
                    self.checked_tensors += 1
                    valid = self.torch.isfinite(tensor)
                    if not bool(valid.all().item()):
                        indices = (~valid).nonzero(as_tuple=False)
                        self.first = dict(module=name, module_type=type(module).__name__,
                            shape=list(tensor.shape), dtype=str(tensor.dtype),
                            nonfinite_elements=int(indices.shape[0]),
                            first_nonfinite_index=indices[0].tolist())
                        break
            return check
        for name, module in self.model.named_modules():
            if (name and (not list(module.children()) or
                          type(module).__name__.endswith(("Attention", "DecoderLayer")))):
                self.handles.append(module.register_forward_hook(hook(name)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def diagnose(torch, server, case, heads, tokenizer, directory, phase):
    """Keep the original failure; never sanitize scores or change weights."""
    report = dict(case=case_identity(case, phase), diagnostic_only=True,
                  hooks_used_inside_formal_timing=False,
                  first_nonfinite_output=None, nonfinite_parameters=[])
    previous_replay_flag = getattr(server, "_finite_diagnostic_replay", False)
    server._finite_diagnostic_replay = True
    try:
        for name, tensor in server.model.named_parameters():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                report["nonfinite_parameters"].append(name)
        with FiniteTracker(torch, server.model) as tracker:
            try:
                server.serve(case, heads, tokenizer)
                report["replay_status"] = "FINITE_ON_REPLAY"
            except Exception as exc:
                report.update(replay_status="FAILED_ON_REPLAY", error_type=type(exc).__name__,
                              reason=str(exc), traceback=traceback.format_exc())
            report.update(first_nonfinite_output=tracker.first,
                          checked_output_tensors=tracker.checked_tensors)
    except Exception as exc:
        report["diagnostic_error"] = type(exc).__name__+": "+str(exc)
    finally:
        server._finite_diagnostic_replay = previous_replay_flag
    dump(directory/"nonfinite_diagnostic.json", report)


def warmup_check(driver, torch, core, e, c, p, o, spec, model, factors,
                 targets, base, contract, directory, banks=None):
    """Exercise six unique heads with full and cached frozen scoring, untimed."""
    from transformers import AutoTokenizer
    work = driver.read(driver.HERE/"input/workload.json")
    tokenizer = AutoTokenizer.from_pretrained(spec["model_dir"], local_files_only=True,
                                              trust_remote_code=False)
    server = core.TraceExecutor(torch, e, c, p, o, spec, model, factors, targets,
                                base, spec["policy"], contract, banks)
    checks = []
    seen = set()
    try:
        for case in work["warmup"]:
            if case["head_id"] in seen:
                continue
            seen.add(case["head_id"])
            for phase in ("full", "cached"):
                dump(directory/"active_score_case.json", case_identity(case, phase))
                before = e.tensor_digest(torch, [(str(i), t) for i, t in
                    enumerate([t for item in server.bank.values()
                               for t in e.cache_tensors(item["cache"])])])
                try:
                    row = server.serve(case, work["heads"], tokenizer)
                except FloatingPointError:
                    diagnose(torch, server, case, work["heads"], tokenizer, directory, phase)
                    raise
                if phase == "cached":
                    after = e.tensor_digest(torch, [(str(i), t) for i, t in
                        enumerate([t for item in server.bank.values()
                                   for t in e.cache_tensors(item["cache"])])])
                    if not row["cache_hit"] or before != after:
                        raise ValueError("Preflight cached hit/template immutability failed")
                elif row["cache_hit"] or not row["cache_admitted"]:
                    raise ValueError("Preflight full miss/cache admission failed")
                actual = None
                if spec["policy"] != "hf_native_cached":
                    actual = e.tensor_digest(torch, [(n, server.params[0][n])
                        for n in targets[case["task"]]])
                    if actual != e.LOCAL_TARGET_HASHES[case["task"]]:
                        raise ValueError("Preflight full mutable-weight hash differs")
                checks.append(dict(case=case_identity(case, phase), scores=row["scores"],
                    cache_hit=row["cache_hit"], mutable_weights_sha256=actual))
                dump(directory/"warmup_scores.json", checks)
        if len(seen) != 6 or len(checks) != 12:
            raise ValueError("Expected six unique frozen heads and twelve scores")
    finally:
        del server
        gc.collect()
        torch.cuda.empty_cache()
    return checks


@contextlib.contextmanager
def record_probe_cases(torch, core, directory):
    """Only the quality-only preflight has per-case context writes/hooks."""
    original = core.TraceExecutor
    class RecordingExecutor(original):
        def serve(self, case, heads, tokenizer):
            if getattr(self, "_finite_diagnostic_replay", False):
                return super().serve(case, heads, tokenizer)
            phase = "trace_or_warmup"
            dump(directory/"active_score_case.json", case_identity(case, phase))
            try:
                return super().serve(case, heads, tokenizer)
            except FloatingPointError:
                diagnose(torch, self, case, heads, tokenizer, directory, phase)
                raise
    core.TraceExecutor = RecordingExecutor
    try:
        yield
    finally:
        core.TraceExecutor = original


def probe_worker(driver, root, name, method):
    import torch
    import bounded_io as bio
    directory = root/"numerical_preflight"/name/method
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status="STARTED", profile=name, method=method, quality_only=True,
                  new_questions=0, accuracy_or_latency_used_to_choose_profile=False)
    try:
        with driver.Telemetry(directory):
            report["observation"] = driver.require_no_observed_foreign(__import__("os").getpgrp())
            core, e, c, p, o = driver.load_core()
            contract = driver.runtime_contract(torch, e, p, name)
            report["runtime_contract"] = contract
            targets, metadata = bio.load_targets(torch, e, driver.HERE/"assets/reference")
            spec = driver.make_spec(method)
            model, factors, base = core.load_base(spec, torch, e, targets,
                    native=method == "native", need_factors=False)
            report["effective_attention_implementation"] = configure_model(model, name)
            bio.trim_heap()
            report["memory_after_load"] = driver.require_host_headroom()
            if method == "reference":
                report["shared_state_before_scoring"] = shared_state(torch,e,model,targets,metadata,
                    directory/"shared_state_before_scoring.json")
            report["warmup_scores"] = warmup_check(driver,torch,core,e,c,p,o,
                spec,model,factors,targets,base,contract,directory)
            dump(directory/"pre_trace_validation.json", report)
            with record_probe_cases(torch,core,directory):
                result = core.execute_trace(directory,spec,spec["policy"],model,factors,
                    targets,base,contract,e,c,p,o,limit=384,write_tag="trace")
            result["target_hashes"] = e.LOCAL_TARGET_HASHES
            driver.audit_records(driver.rows(directory/"trace_requests.jsonl"),
                driver.read(driver.HERE/"input/workload.json")["check"],result,method)
            if method == "reference":
                report["shared_state_after_scoring"] = shared_state(torch,e,model,targets,metadata,
                    directory/"shared_state_after_scoring.json")
            report.update(status="COMPLETED", trace_report=result)
        samples = driver.rows(directory/"telemetry.jsonl")
        if not samples or any(x["observation"]["foreign_pids"] for x in samples):
            raise ValueError("Missing telemetry or an observed foreign GPU client")
    except FloatingPointError as exc:
        report.update(status="NUMERICAL_FAILURE", error_type=type(exc).__name__,
                      reason=str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        report.update(status="ERROR", error_type=type(exc).__name__,
                      reason=str(exc), traceback=traceback.format_exc())
    finally:
        dump(directory/"report.json", report)
    return 0 if report["status"] == "COMPLETED" else 2


def eligible(report):
    if report["status"] == "NUMERICAL_FAILURE":
        return False
    if report["status"] != "COMPLETED":
        raise RuntimeError("Non-numerical preflight failure; see its report: "+str(report.get("reason")))
    if report["trace_report"]["requests"] != 384 or len(report["warmup_scores"]) != 12:
        raise ValueError("Incomplete preflight cannot select a numerical profile")
    return True


def select_profile(driver, root, timeout):
    selected_path = root/"numerical_preflight/selection.json"
    if selected_path.exists():
        selected = read(selected_path)
        for method in ("reference", "native"):
            report = read(root/"numerical_preflight"/selected["profile"]/method/"report.json")
            if not eligible(report) or report["runtime_contract"] != selected["runtime_contract"]:
                raise ValueError("Saved numerical selection differs from completed preflight")
        return selected
    attempts = []
    for name in PROFILE_ORDER:
        accepted = True
        contracts = []
        for method in ("reference", "native"):
            directory = root/"numerical_preflight"/name/method
            report_path = directory/"report.json"
            if not report_path.exists():
                if directory.exists():
                    raise RuntimeError("Interrupted numerical probe retained. Start a fresh run; do not overwrite it: "+str(directory))
                log = root/"logs"/("preflight_"+name+"_"+method+".log")
                log.parent.mkdir(parents=True,exist_ok=True)
                print("PREFLIGHT:", name, method, "(6 heads + 384 quality-only requests)", flush=True)
                with log.open("w") as stream:
                    process = subprocess.Popen([sys.executable,str(driver.HERE/"run_crossdevice.py"),
                        "_probe","--result-dir",str(root),"--numerical-profile",name,
                        "--probe-method",method],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        process.wait(timeout=timeout)
                    except (subprocess.TimeoutExpired,KeyboardInterrupt):
                        import os,signal
                        os.killpg(process.pid,signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        raise
                dump(root/"process_exits"/("preflight_"+name+"_"+method+".json"),
                     dict(pid=process.pid,returncode=process.returncode,log=str(log)))
                if not report_path.exists():
                    raise RuntimeError("Preflight worker stopped before a report; see "+str(log))
                expected_exit = 0 if read(report_path)["status"] == "COMPLETED" else 2
                if process.returncode != expected_exit:
                    raise RuntimeError("Preflight process exit/report disagree; see "+str(log))
            report = read(report_path)
            attempts.append(dict(profile=name,method=method,status=report["status"],
                                 report=str(report_path.relative_to(root))))
            dump(root/"numerical_preflight/attempts.json",attempts)
            if not eligible(report):
                accepted = False
                print("PREFLIGHT_NUMERICAL_FAILURE:",name,method,"(retained in archive)",flush=True)
                break
            contracts.append(report["runtime_contract"])
        if accepted:
            if contracts[0] != contracts[1]:
                raise ValueError("Reference/native preflight arithmetic contracts differ")
            selected = dict(profile=name,runtime_contract=contracts[0],
                selection_rule="First profile in registered order passing both finite/integrity preflights",
                profile_order=list(PROFILE_ORDER),accuracy_or_latency_used=False,
                quality_reference_rows=f"numerical_preflight/{name}/reference/trace_requests.jsonl",
                native_preflight_rows=f"numerical_preflight/{name}/native/trace_requests.jsonl",
                timing_not_pooled_with_rtx=True,new_questions=0)
            dump(selected_path,selected)
            print("NUMERICAL_PROFILE_SELECTED:",name,flush=True)
            return selected
    raise FloatingPointError("All registered profiles failed finite scoring; diagnostics retained. No corrections were built.")
