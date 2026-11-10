#!/usr/bin/env python3
"""Replay a controlled LoRA switch-history depth sweep from the paper runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import tarfile
import uuid


ROOT = Path(__file__).resolve().parent
TASKS = ("gsm8k", "dialogsum")
DEPTHS = (0, 2, 8, 32, 128)
FINAL_SOURCE = "qnli"
RESIDENT = "confirm_resident_host_copy"
INCREMENTAL = "confirm_r0_incremental"
FIELDS = (
    "task", "return_cycles", "switches", "final_source", "distinct_prompts",
    "canonical_weights", "exact_sequences", "divergent_sequences", "divergent_rate",
    "median_first_differing_token", "nonfinite_logits", "length_limit",
    "parsed_answer_changes", "unparsed_answers", "correct", "correct_to_incorrect",
    "incorrect_to_correct", "mean_rouge_l_f1", "mean_rouge_l_f1_delta",
)


def load_modules(task):
    directory = ROOT / "generation" / task
    # GSM8K and DialogSum ship separate modules with the same import names.
    sys.path[:0] = [str(directory), str(directory / "runtime/vendor/runtime/vendor")]
    import generation_protocol as gp
    import run_generation as runner

    if gp.TASK != task or any(gp.peer(depth) != FINAL_SOURCE for depth in DEPTHS[1:]):
        raise ValueError("The frozen peer schedule does not yield the same incoming source")
    runner.check_release()
    return gp, runner


def committed(gp, source, phase, relative):
    marker = gp.read_json(source / "completed" / f"{phase}.json")
    path = source / relative
    if gp.sha(path) != marker["outputs"].get(relative):
        raise ValueError(f"Original completed output changed: {path}")
    return path


def read_stage(gp, source, key, workload):
    job = next(job for job in gp.schedule("confirm") if job["key"] == key)
    requests = committed(gp, source, key, f"stages/{key}/requests.jsonl")
    report = gp.read_json(committed(gp, source, key, f"stages/{key}/report.json"))
    rows = gp.rows(requests)
    gp.audit_stage(job, workload, rows, report)
    return rows, report, gp.sha(requests)


def check_paper_trace(archived, published):
    # The public export remaps token IDs. IDs, timing, and token counts are unchanged.
    fields = ("id", "cycle", "from_task", "stop_reason", "ttft_ms", "request_ms")
    if len(archived) != len(published) or any(
        any(raw[key] != public[key] for key in fields)
        or len(raw["generated_ids"]) != len(public["tokens"])
        for raw, public in zip(archived, published)
    ):
        raise ValueError("The run is not the original confirmation recorded in the paper")


def load_source(gp, source):
    config = gp.read_json(source / "run_config.json")
    summary = gp.read_json(source / "confirm_summary.json")
    workload_path = committed(gp, source, "data", "inputs/workload.json")
    workload = gp.read_json(workload_path)
    cases = workload["test"]
    if (config["protocol"] != gp.PROTOCOL_VERSION or summary["status"] != "COMPLETED"
            or len(cases) != 128 or len({case["id"] for case in cases}) != 128
            or len(workload["warmup"]) != len(gp.PEERS)
            or any(gp.identity(case["prompt_ids"]) != case["prompt_sha256"]
                   for case in cases + workload["warmup"])):
        raise ValueError(f"Invalid frozen paper inputs: {source}")
    resident, report, resident_sha = read_stage(gp, source, RESIDENT, workload)
    incremental, inc_report, incremental_sha = read_stage(gp, source, INCREMENTAL, workload)
    with gzip.open(ROOT / "evidence" / f"{gp.TASK}.json.gz", "rt", encoding="utf-8") as handle:
        evidence = json.load(handle)
    for rows, method, block in ((resident, "host_copy", None), (incremental, "incremental", 0)):
        stage = next(s for s in evidence["stages"] if s["method"] == method and s["block"] == block)
        check_paper_trace(rows, stage["rows"])
    contract = gp.read_json(source / "runtime_contract.json")
    if any(contract[key] != value for key, value in evidence["runtime"].items()):
        raise ValueError("Recorded hardware/software differs from the paper")
    task = gp.TASK
    ref_meta = gp.read_json(committed(gp, source, f"merge_{task}", f"artifacts/references/{task}.json"))
    canonical = ref_meta["tensor_sha256"]
    digest_key = "canonical_gsm8k_digest" if task == "gsm8k" else "canonical_target_digest"
    if (report[digest_key] != canonical or inc_report[digest_key] != canonical
            or resident[0]["weight_digest"] != canonical):
        raise ValueError("Archived stages disagree on the canonical FP16 target")
    anchors = {depth: incremental[depth - 1] for depth in DEPTHS[1:]}
    if any(row["from_task"] != FINAL_SOURCE or row["cycle"] != depth
           or len(row.get("weight_digest") or "") != 64 for depth, row in anchors.items()):
        raise ValueError("An original switch-history anchor is missing")
    source_hashes = dict(
        workload_sha256=gp.sha(workload_path), resident_sha256=resident_sha,
        original_incremental_sha256=incremental_sha, canonical_weight_sha256=canonical,
        target_file_sha256=ref_meta["file_sha256"], adapter_sha256=ref_meta["adapter_sha256"],
        contract_sha256=gp.identity(contract),
        release_sha256=gp.sha(ROOT / "generation" / task / "MANIFEST_SHA256.json"),
        sweep_code_sha256=gp.sha(Path(__file__)),
    )
    return config, workload, resident, report, anchors, source_hashes


def check_assets(gp, source, config, source_hashes, hash_shards):
    task = gp.TASK
    assets = gp.read_json(ROOT / "generation" / task / "paper_assets.json")
    model = Path(config["model_dir"])
    if not model.is_absolute():
        raise ValueError("The original model path must be absolute")
    model_pins = {model / name: digest for name, digest in assets["model_files_sha256"].items()}
    paths = [
        f"artifacts/references/{name}.{extension}"
        for name in ("base", *gp.PEERS) for extension in ("json", "safetensors")
    ] + [
        f"trained/lora/{name}/{file}"
        for name in gp.PEERS for file in ("adapter_config.json", "adapter_model.safetensors")
    ]
    peer_pins = {source / name: assets["reference_run_files_sha256"][name] for name in paths}
    merge = gp.read_json(source / "completed" / f"merge_{task}.json")["outputs"]
    train = gp.read_json(source / "completed" / f"train_{task}.json")["outputs"]
    target = f"artifacts/references/{task}.safetensors"
    adapter = f"trained/lora/{task}/adapter_model.safetensors"
    if (merge[target] != source_hashes["target_file_sha256"]
            or train[adapter] != source_hashes["adapter_sha256"]):
        raise ValueError("Target merge and training commitments differ from the reference metadata")
    target_pins = {source / name: digest for name, digest in (
        (target, merge[target]), ("artifacts/references/metadata.json", merge["artifacts/references/metadata.json"]),
        (adapter, train[adapter]), (f"trained/lora/{task}/adapter_config.json",
                                    train[f"trained/lora/{task}/adapter_config.json"]),
    )}
    for path, digest in (model_pins | peer_pins | target_pins).items():
        if not path.is_file():
            raise FileNotFoundError(f"Original model/LoRA asset is missing: {path}")
        # The unchanged Session verifies the multi-GiB reference files while loading them.
        if path.stat().st_size < 2**28 or hash_shards and path in model_pins:
            if gp.sha(path) != digest:
                raise ValueError(f"Original asset hash differs: {path}")
    for name in ("base", *gp.PEERS):
        relative = f"artifacts/references/{name}.safetensors"
        record = gp.read_json(source / f"artifacts/references/{name}.json")
        if record["file_sha256"] != assets["reference_run_files_sha256"][relative]:
            raise ValueError(f"Original peer reference commitment differs: {name}")
    return model


def compare_request(gp, case, reference, tokens, reason, text, depth):
    difference = gp.first_difference(reference["generated_ids"], tokens)
    result = dict(id=case["id"], prompt_sha256=case["prompt_sha256"], return_cycles=depth,
                  generated_ids=tokens, reference_ids=reference["generated_ids"],
                  first_differing_token=difference, stop_reason=reason)
    if gp.TASK == "gsm8k":
        prediction = None if reason == "nonfinite_logits" else gp.numeric_answer(text)
        result.update(prediction=prediction, reference_prediction=reference["prediction"],
                      gold=case["gold"], correct=prediction == case["gold"],
                      reference_correct=reference["correct"])
    else:
        quality, _ = gp.summary_quality(text, case["references"], reason)
        result.update(rouge_l_f1=quality, reference_rouge_l_f1=reference["rouge_l_f1"])
    return result


def summarize_depth(task, depth, rows, canonical_weights):
    differences = [row["first_differing_token"] for row in rows if row["first_differing_token"] is not None]
    result = dict(task=task, return_cycles=depth, switches=2 * depth,
                  final_source=FINAL_SOURCE if depth else None, distinct_prompts=len(rows),
                  canonical_weights=canonical_weights, exact_sequences=len(rows) - len(differences),
                  divergent_sequences=len(differences), divergent_rate=len(differences) / len(rows),
                  median_first_differing_token=median(differences) if differences else None,
                  nonfinite_logits=sum(row["stop_reason"] == "nonfinite_logits" for row in rows),
                  length_limit=sum(row["stop_reason"] == "length_limit" for row in rows))
    if task == "gsm8k":
        result.update(
            parsed_answer_changes=sum(row["prediction"] != row["reference_prediction"] for row in rows),
            unparsed_answers=sum(row["prediction"] is None for row in rows),
            correct=sum(row["correct"] for row in rows),
            correct_to_incorrect=sum(row["reference_correct"] and not row["correct"] for row in rows),
            incorrect_to_correct=sum(not row["reference_correct"] and row["correct"] for row in rows),
        )
    else:
        result.update(mean_rouge_l_f1=mean(row["rouge_l_f1"] for row in rows),
                      mean_rouge_l_f1_delta=mean(
                          row["rouge_l_f1"] - row["reference_rouge_l_f1"] for row in rows))
    return result


def advance(state, gp, current, depth):
    for cycle in range(current + 1, depth + 1):
        state.switch(gp.peer(cycle))
        state.torch.cuda.synchronize()
        state.switch(gp.TASK)
        state.torch.cuda.synchronize()


def replay(gp, gw, state, tokenizer, workload, resident, report, anchors, port, attempt):
    eos = state.model.generation_config.eos_token_id
    eos_ids = frozenset(([eos] if isinstance(eos, int) else eos or []) + [tokenizer.eos_token_id])
    if (None in eos_ids or sorted(eos_ids) != report["eos_ids"] or
            state.model.config.vocab_size != report["vocab_size"]):
        raise ValueError("Model EOS or vocabulary differs from the recorded reference")
    for peer, case in zip(gp.PEERS, workload["warmup"]):
        state.switch(peer)
        state.switch(gp.TASK)
        gw.gen_tokens(state.torch, state.model, "cuda:0", state.tail_arg, case["prompt_ids"], eos_ids, limit=8)
    state.restore_initial()
    canonical = state.digest()
    current = 0
    summaries = []
    state_checks = []
    changed = set()
    with (attempt / "requests.jsonl").open("w", encoding="utf-8") as trace:
        for depth in DEPTHS:
            advance(state, gp, current, depth)
            current = depth
            before = state.digest()
            anchor = anchors.get(depth)
            expected = anchor["weight_digest"] if anchor else canonical
            if before != expected:
                raise ValueError(f"Replayed mutable weights differ from original at {depth} return cycles")
            print(f"DEPTH {gp.TASK} {depth}: evaluating {len(resident)} frozen prompts", flush=True)
            rows = []
            for index, (case, reference) in enumerate(zip(workload["test"], resident), 1):
                tokens, reason, _, _, cache = gw.gen_tokens(
                    state.torch, state.model, "cuda:0", state.tail_arg, case["prompt_ids"], eos_ids,
                    gp.MAX_NEW_TOKENS, allow_nonfinite=depth > 0,
                )
                del cache
                if depth == 0 or case["id"] == anchor["id"]:
                    old = anchor if depth else reference
                    if tokens != old["generated_ids"] or reason != old["stop_reason"]:
                        raise ValueError(f"Original generation did not replay: {case['id']} at depth {depth}")
                text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                row = compare_request(gp, case, reference, tokens, reason, text, depth)
                rows.append(row)
                if row["first_differing_token"] is not None:
                    changed.add(case["id"])
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if index % 16 == 0:
                    trace.flush()
                    port.require_no_observed_foreign(os.getpgrp())
                    print(f"ACTIVE {gp.TASK} {depth}: {index}/{len(resident)}", flush=True)
            after = state.digest()
            if after != before:
                raise ValueError(f"Mutable weights changed during the depth {depth} prompt group")
            state_checks.append(dict(return_cycles=depth, before=before, after=after,
                                     original_trace=expected, anchor_prompt=anchor["id"] if anchor else None))
            summaries.append(summarize_depth(gp.TASK, depth, rows, before == canonical))
    return summaries, state_checks, len(changed)


def check_complete(gp, target, source_hashes):
    record = gp.read_json(target / "COMPLETE.json")
    if (record["sources"] != source_hashes
            or set(record["outputs"]) != {"requests.jsonl", "summary.json", "telemetry.jsonl"}
            or any(gp.sha(target / name) != digest for name, digest in record["outputs"].items())):
        raise ValueError(f"Completed sweep changed: {target}. Use a separate output directory.")


def gpu_run(gp, runner, source, output, loaded, idle_timeout):
    config, workload, resident, report, anchors, source_hashes = loaded
    model = check_assets(gp, source, config, source_hashes, hash_shards=False)
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (source, model.resolve())):
        raise ValueError("Use a separate sweep output directory")
    target = output / gp.TASK
    if target.exists():
        check_complete(gp, target, source_hashes)
        print(f"REUSE {gp.TASK}: {target}", flush=True)
        return
    attempt = output / "_attempts" / f"{gp.TASK}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    os.environ.update(runner.worker_env())
    from transformers import AutoTokenizer
    import generation_worker as gw

    check_assets(gp, source, config, source_hashes, hash_shards=True)
    runner.wait_gpu(idle_timeout)
    gw.configure_paths(config)
    port = gw.configure_port()
    with port.Telemetry(attempt):
        port.require_no_observed_foreign(os.getpgrp())
        state = gw.ReturnState(source, port, "incremental")
        if state.digest() != source_hashes["canonical_weight_sha256"]:
            raise ValueError("Loaded mutable weights differ from the original merged target")
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        summaries, checks, changed = replay(
            gp, gw, state, tokenizer, workload, resident, report, anchors, port, attempt,
        )
        port.require_no_observed_foreign(os.getpgrp())
    samples = gp.rows(attempt / "telemetry.jsonl")
    if not samples or any(
        sample["observation"]["backend"] != "nvml" or
        not sample["observation"]["process_enumeration_verified"] or
        sample["observation"]["foreign_pids"]
        for sample in samples
    ):
        raise RuntimeError("GPU isolation failed or could not be verified")
    gp.write_json(attempt / "summary.json", dict(
        task=gp.TASK, depths=summaries, weight_checks=checks,
        distinct_prompts_ever_divergent=changed, telemetry_samples=len(samples),
        gpu_isolation_verified=True, sources=source_hashes, boot_id=gw.boot_id(),
    ))
    gp.write_json(attempt / "COMPLETE.json", dict(sources=source_hashes, outputs={
        name: gp.sha(attempt / name) for name in ("requests.jsonl", "summary.json", "telemetry.jsonl")
    }))
    attempt.rename(target)
    print(f"COMPLETED {gp.TASK}: {target}", flush=True)


def pack(output):
    # Both protocols implement the same JSON and SHA utilities.
    gp, _ = load_modules("gsm8k")
    summaries = {}
    for task in TASKS:
        directory = output / task
        summary = gp.read_json(directory / "summary.json")
        check_complete(gp, directory, summary["sources"])
        if (summary["task"] != task or not summary["gpu_isolation_verified"]
                or tuple(row["return_cycles"] for row in summary["depths"]) != DEPTHS
                or any(row["distinct_prompts"] != 128 for row in summary["depths"])):
            raise ValueError(f"Incomplete or invalid depth sweep: {directory}")
        summaries[task] = summary
    gp.write_json(output / "summary.json", dict(status="COMPLETED", tasks=summaries))
    with (output / "summary.csv.partial").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for summary in summaries.values():
            writer.writerows(summary["depths"])
    (output / "summary.csv.partial").replace(output / "summary.csv")
    archive = output / "FTD_ACL_SwitchHistory_Results.tar.gz"
    partial = archive.with_name(archive.name + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=3) as stream:
        for name in ("summary.csv", "summary.json"):
            stream.add(output / name, arcname=name)
        for task in TASKS:
            for name in ("summary.json", "requests.jsonl", "telemetry.jsonl", "COMPLETE.json"):
                stream.add(output / task / name, arcname=f"{task}/{name}")
    partial.replace(archive)
    print(f"SEND_THIS_ARCHIVE: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "audit", "check", "run", "pack"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--idle-timeout", type=int, default=8 * 3600)
    args = parser.parse_args()
    if args.command == "plan":
        for depth in DEPTHS:
            print(f"{depth:3d} return cycles | {2 * depth:3d} switches | {FINAL_SOURCE if depth else 'resident'}")
        return
    if args.command in ("run", "pack") and args.output is None:
        parser.error("--output is required")
    if args.command == "pack":
        pack(args.output.resolve())
        return
    if args.task is None or args.run_dir is None or args.idle_timeout < 1:
        parser.error("--task, --run-dir and a positive --idle-timeout are required")
    source = args.run_dir.resolve(strict=True)
    gp, runner = load_modules(args.task)
    loaded = load_source(gp, source)
    if args.command == "run":
        gpu_run(gp, runner, source, args.output.resolve(), loaded, args.idle_timeout)
    else:
        if args.command == "check":
            check_assets(gp, source, loaded[0], loaded[-1], hash_shards=False)
        print(f"VERIFIED {args.task}: original 128 prompts, resident trace, and {len(loaded[4])} history anchors")


if __name__ == "__main__":
    main()
