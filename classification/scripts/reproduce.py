#!/usr/bin/env python3
"""Reconstruct main-paper measurements from the released request and Zstd records."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from paper_protocol import (
    CAP, FREQUENCY_METHODS, KV_CAP, METHODS, ROOT, SWITCHES, TASKS,
    confirmation_jobs, frequency_jobs, frequency_order,
)

DATA = ROOT / "data"
PARTS = ("tokenize_ms", "parameter_ms", "decision_ms", "score_ms", "admission_ms")
MEMORY_FIELDS = ("fixed_tensor_bytes", "retained_patch_bytes", "allocator_peak_bytes")
SCORE_LABELS = {
    "mrpc": {"equivalent", "not_equivalent"},
    "qnli": {"entailment", "not_entailment"},
    "agnews": {"world", "sports", "business", "science"},
}
ZSTD_UNIT_COUNTS = {
    "canonical_mutable_raw_zstd": 756,
    "canonical_mutable_xor_base_zstd": 756,
    "direct_patch_zstd": 1512,
    "shared_patch_zstd": 756,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify_manifest(root, manifest):
    for name, expected in read(manifest).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked release file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Release file digest differs: {name}")


def percentile95(values):
    values = sorted(values)
    at = 0.95 * (len(values) - 1)
    lower = math.floor(at)
    upper = math.ceil(at)
    return values[lower] + (at - lower) * (values[upper] - values[lower])


def stage_metrics(rows, report):
    latency = [row["request_ms"] for row in rows]
    return dict(
        mean_ms=mean(latency),
        p95_ms=percentile95(latency),
        parameter_mean_ms=mean(row["parameter_ms"] for row in rows),
        score_mean_ms=mean(row["score_ms"] for row in rows),
        correct=sum(row["prediction"] == row["gold"] for row in rows),
        cache_hits=sum(row["cache_hit"] for row in rows),
        switches=report["switches"],
        **{field: report[field] for field in MEMORY_FIELDS},
    )


def audit_trace(stage, order, reference, expected_switches):
    job = stage["job"]
    rows = stage["records"]
    report = stage["report"]
    method = job["method"]
    if len(rows) != 384 or report["requests"] != 384:
        raise ValueError(f"Incomplete trace: {job}")
    if [row["query"] for row in rows] != [row["query"] for row in order]:
        raise ValueError(f"Workload order differs: {job}")
    if report["logical_cap_bytes"] != CAP or report["retained_kv_cap_bytes"] != KV_CAP:
        raise ValueError("Logical cap differs")
    if report["logical_live_max_bytes"] > CAP or not stage["warmup_immutable"]:
        raise ValueError("Logical cap or warmup check failed")
    if method != "native" and not stage["final_canonical_weight_hash_matches"]:
        raise ValueError("Final canonical weight hash failed")
    if method in ("shared", "direct", "node"):
        expected = dict(projection_checks=1512, grouped_paths=24, all_projections_equal=True)
        if report["mutable_projections"] != 252 or stage["admission"] != expected:
            raise ValueError("Full projection/path admission missing")
    if not stage["samples"] or any(verified is not True or count for verified, count in stage["samples"]):
        raise ValueError("Formal stage failed GPU observation checks")
    current = TASKS[0]
    switches = 0
    maximum_score_difference = 0.0
    for position, row in enumerate(rows):
        ref = reference[row["query"]]
        if row["position"] != position or any(row[k] != ref[k] for k in ("task", "gold")):
            raise ValueError("Request identity changed")
        parts = [row[key] for key in PARTS]
        if any(not math.isfinite(x) or x < 0 for x in parts) or not math.isclose(
            sum(parts), row["request_ms"], abs_tol=1e-8
        ):
            raise ValueError("Request timing components differ")
        scores = row["scores"]
        if set(scores) != SCORE_LABELS[row["task"]] or any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("Candidate score schema differs")
        if row["prediction"] != max(sorted(scores), key=scores.get):
            raise ValueError("Prediction does not follow the score tie rule")
        if row["prediction"] != ref["prediction"] or row["cache_hit"] != ref["cache_hit"]:
            raise ValueError("Recorded prediction/cache result differs from the reference")
        if method != "native" and scores != ref["scores"]:
            raise ValueError("Canonical candidate score mismatch")
        maximum_score_difference = max(
            maximum_score_difference, *(abs(scores[label] - ref["scores"][label]) for label in scores)
        )
        switches += int(current != row["task"])
        current = row["task"]
        if row["total_switches"] != switches or not 0 <= row["cache_bytes"] <= KV_CAP:
            raise ValueError("Switch or cache accounting differs")
    if switches != report["switches"] or switches != expected_switches:
        raise ValueError("Switch count differs")
    result = stage_metrics(rows, report)
    if result["correct"] != 346 or result["cache_hits"] != 378:
        raise ValueError("Recorded quality/cache count differs")
    return dict(job=job, max_score_difference_vs_reference=maximum_score_difference, **result)


def aggregate(items):
    return dict(
        mean_ms=mean(x["mean_ms"] for x in items),
        mean_block_p95_ms=mean(x["p95_ms"] for x in items),
        parameter_mean_ms=mean(x["parameter_mean_ms"] for x in items),
        score_mean_ms=mean(x["score_mean_ms"] for x in items),
        correct_each_block=[x["correct"] for x in items],
        cache_hits_each_block=[x["cache_hits"] for x in items],
        **{k: max(x[k] for x in items) for k in MEMORY_FIELDS},
    )


def complete_boot_blocks(stages, boot_for_job, methods, regime):
    """Use only blocks in which every required method ran on the same boot."""
    by_block = defaultdict(list)
    for item in stages:
        if item["job"].get("regime") == regime:
            by_block[item["job"]["block"]].append(item)
    by_boot = defaultdict(list)
    for block, items in sorted(by_block.items()):
        boots = {boot_for_job[item["job"]["tag"], regime] for item in items}
        if len(boots) == 1 and {item["job"]["method"] for item in items} == set(methods):
            by_boot[boots.pop()].append(block)
    return [
        dict(
            boot_id=boot, regime=regime, complete_three_method_blocks=blocks,
            mean_ms={
                method: mean(item["mean_ms"] for block in blocks for item in by_block[block]
                             if item["job"]["method"] == method)
                for method in methods
            },
        )
        for boot, blocks in by_boot.items()
    ]


def reconstruct_serving(provenance):
    cohort = list(records(DATA / "reference.jsonl.gz"))
    reference = {row["query"]: row for row in cohort}
    if len(cohort) != 384 or set(reference) != set(range(384)) or Counter(
        row["task"] for row in cohort
    ) != {task: 128 for task in TASKS}:
        raise ValueError("Reference cohort differs")
    orders = {None: cohort, **{regime: frequency_order(cohort, regime) for regime in SWITCHES}}
    results = {}
    for name, plan in (("confirmation", confirmation_jobs()), ("frequency", frequency_jobs())):
        raw = list(records(DATA / (name + ".jsonl.gz")))
        if [stage["job"] for stage in raw] != plan:
            raise ValueError(f"Committed stage order differs: {name}")
        samples = sum(len(stage["samples"]) for stage in raw)
        if samples != provenance["formal_worker_gpu_samples"][name]:
            raise ValueError(f"GPU observation count differs: {name}")
        boot_map = {}
        stages = []
        for stage in raw:
            job = stage["job"]
            regime = job.get("regime")
            stages.append(audit_trace(stage, orders[regime], reference, SWITCHES.get(regime, 136)))
            boot_map[job["tag"], regime] = stage["boot"]
        results[name] = stages, boot_map
        print(f"{name}: {len(stages)} stages, {384 * len(stages):,} requests, {samples:,} worker GPU samples")
    fresh, fresh_boots = results["confirmation"]
    freq, freq_boots = results["frequency"]
    return (
        dict(status="FRESH_CONFIRMATION_COMPLETED", stages=fresh, formal_requests=19200,
             aggregates={m: aggregate([x for x in fresh if x["job"]["method"] == m]) for m in METHODS}),
        dict(status="LOCAL_AUDIT_PASSED", stages=freq, formal_stages=54,
             formal_requests=20736, distinct_questions=384,
             aggregates={
                 regime: {m: aggregate([x for x in freq if x["job"]["regime"] == regime and
                                       x["job"]["method"] == m]) for m in FREQUENCY_METHODS}
                 for regime in SWITCHES
             },
             formal_stage_counts_by_boot=dict(Counter(freq_boots.values())),
             complete_same_boot_block_sensitivity=[
                 group for regime in SWITCHES
                 for group in complete_boot_blocks(freq, freq_boots, FREQUENCY_METHODS, regime)
             ]),
        {tag: boot for (tag, _), boot in fresh_boots.items()},
    )


def reconstruct_zstd():
    framing = read(DATA / "zstd_framing.json")
    groups = {
        key: dict(value, raw_tensor_bytes=0, zstd_frame_bytes=0,
                  frame_prefix_bytes=0, cpu_restore_median_sum_ms=0.0, units=0, blocks=0)
        for key, value in framing["groups"].items()
    }
    seen = set()
    for unit in records(DATA / "zstd_units.jsonl.gz"):
        key = unit["key"]
        measurement = unit["measurement"]
        if key in seen or not measurement["raw_bytes_identical"]:
            raise ValueError("Duplicate or unverified Zstd unit")
        seen.add(key)
        group = groups[key.split("/")[0]]
        if measurement["frame_prefix_bytes"] != 8 * measurement["blocks"]:
            raise ValueError("Frame-length accounting differs")
        group["units"] += 1
        group["raw_tensor_bytes"] += measurement["raw_bytes"]
        for field in ("zstd_frame_bytes", "frame_prefix_bytes", "cpu_restore_median_sum_ms", "blocks"):
            group[field] += measurement[field]
    if {name: group["units"] for name, group in groups.items()} != ZSTD_UNIT_COUNTS:
        raise ValueError("Zstd unit coverage differs")
    for group in groups.values():
        group["computed_container_bytes"] = sum(group[key] for key in (
            "zstd_frame_bytes", "frame_prefix_bytes", "container_prefix_and_header_bytes"
        ))
        group["coded_bytes_including_side_metadata"] = group["computed_container_bytes"] + group["side_metadata_bytes"]
        group["coded_bytes_plus_required_reference"] = (
            group["coded_bytes_including_side_metadata"] + group["required_immutable_reference_file_bytes"]
        )
        group["raw_recovery_verified"] = True
    print(f"Zstd: {len(seen):,} measured, byte-verified units")
    return dict(status="LOSSLESS_BASELINE_COMPLETED", groups=groups, configuration=framing["configuration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build"))
    args = parser.parse_args()
    verify_manifest(ROOT, ROOT / "MANIFEST_SHA256.json")
    verify_manifest(ROOT / "runtime", ROOT / "runtime/MANIFEST.json")
    provenance = read(DATA / "provenance.json")
    fresh, frequency, boots = reconstruct_serving(provenance)
    codec = reconstruct_zstd()
    out = args.output.resolve()
    for name, value in (
        ("fresh_confirmation_summary.json", fresh), ("switch_frequency_audit.json", frequency),
        ("fresh_stage_boots.json", boots), ("cpu_zstd_summary.json", codec),
    ):
        write(out / "evidence" / name, value)

    from render_paper import render
    render(out)
    for expected in (DATA / "expected").glob("*.tex"):
        if (out / "tables" / expected.name).read_bytes() != expected.read_bytes():
            raise ValueError(f"Recomputed table differs from the paper: {expected.name}")
    write(out / "audit.json", dict(
        status="PAPER_TABLES_RECONSTRUCTED", confirmation_stages=50, frequency_stages=54,
        unique_questions=384, canonical_correct=346, zstd_units=3780, tables_matched=4,
        stage_mean_definition="Mean across stage means",
        tail_definition="Mean across within-stage interpolated p95 values",
        data_scope="Reconstruction from archived measurements",
    ))
    print(f"Four tables match the paper. Tables, figure and audit are in {out}")


if __name__ == "__main__":
    main()

