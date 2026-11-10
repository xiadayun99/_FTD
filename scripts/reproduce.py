"""Reconstruct the ACL generation tables from recorded measurements on a CPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from reproduce_switch_history import rebuild as rebuild_switch_history

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVIDENCE = ROOT / "evidence"
TASKS = {"gsm8k": ("GSM8K", "Gsm"), "dialogsum": ("DialogSum", "Dialog")}
METHODS = {
    "host_copy": "Host copy", "incremental": "Incremental", "direct": "Direct",
    "shared": "Shared", "native": "HF Native",
}
EXACT = {"host_copy", "direct", "shared"}
AUDIT_CYCLES = {1, 2, 8, 32, 64, 128}
TIMINGS = ("preceding_outbound_switch_ms", "switch_ms", "ttft_ms", "request_ms")
MEMORY = ("peak_allocated_bytes", "peak_reserved_bytes")
FORMATS = {
    "generation_main": ("@{}lrrcrrrr@{}", [
        r" & \multicolumn{2}{c}{Exact sequence} & Quality & TTFT & End-to-end & Bank & Peak \\",
        r"\cmidrule(lr){2-3}",
        r"Method & Block 1 & Block 2 & (B1 / B2) & (ms) & (s) & (GiB) & (GiB) \\",
    ]),
    "paired_timing": ("@{}llrrrr@{}", [
        r" & & \multicolumn{3}{c}{Mean difference} & p95 diff. \\",
        r"\cmidrule(lr){3-5}",
        r"Task & Block & Switch & TTFT & E2E & E2E \\",
    ]),
    "generation_by_block": ("@{}llrrrrrrr@{}", [
        r" & & Outbound & Return & \multicolumn{2}{c}{TTFT (ms)} & \multicolumn{2}{c}{End-to-end (s)} & Reserved \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Block & Method & (ms) & (ms) & Mean & p95 & Mean & p95 & peak (GiB) \\",
    ]),
    "generation_build": ("@{}lrrrrr@{}", [
        r" & \multicolumn{2}{c}{Direct (s)} & \multicolumn{3}{c}{Shared (s)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Task & Encode & Write & Setup & Calibrate & Write \\",
    ]),
}
COMMENT = "% Recomputed from the frozen formal generation traces."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_release():
    entries = json.loads((ROOT / "MANIFEST_SHA256.json").read_text())
    require(bool(entries), "Empty release manifest")
    for name, digest in entries.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),
                f"Missing or unsafe release path: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Modified release file: {name}")


def p95(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def gib(value):
    return value / 2**30


def percent_saved(old, new):
    return f"{100 * (1 - new / old):.1f}"


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return index
    return min(len(left), len(right)) + 1 if len(left) != len(right) else None


def quality(task, row):
    if task == "gsm8k":
        return row["prediction"] == row["gold"]
    counts = row["rouge_l_counts"]
    require(len(counts) == 3 and all(0 <= 2 * lcs <= size and size > 0 for lcs, size in counts),
            "Invalid DialogSum ROUGE-L count fields")
    if row["stop_reason"] == "nonfinite_logits":
        return 0.0
    return mean(2 * lcs / size for lcs, size in counts)


def audit_workload(path, task, published):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    require(data["schema"] == 1 and data["task"] == task, f"Unexpected evidence in {path.name}")
    stages = {(stage["method"], stage["block"]): stage for stage in data["stages"]}
    residents = {("host_copy", None), ("native", None)}
    expected = {(method, block) for method in METHODS for block in (0, 1)} | residents
    require(set(stages) == expected and len(data["stages"]) == 12, f"{task}: stage coverage differs")
    records = {key: {r["id"]: r for r in stage["rows"]} for key, stage in stages.items()}
    reference = records["host_copy", None]
    native_resident = records["native", None]
    require(len(reference) == 128, f"{task}: test cohort size differs")
    metrics, divergences = {}, []
    for (method, block), stage in stages.items():
        rows = records[method, block]
        order = list(reversed(reference)) if block == 1 else list(reference)
        require(len(stage["rows"]) == len(order) and list(rows) == order,
                f"{task}: inputs or order differ for {method}, block {block}")
        if published:
            require(all(row["stop_reason"] == "eos" for row in rows.values()), "Non-EOS published row")
        for counter in MEMORY:
            require(max(row[counter] for row in rows.values()) == stage[counter],
                    f"{task}: {counter} differs for {method}, block {block}")
        require(stage["telemetry_samples"] > 0, f"{task}: missing process observations")
        if block is None:
            continue
        exact = 0
        for cycle, row in enumerate(rows.values(), 1):
            ref = reference[row["id"]]
            require(row["cycle"] == cycle and row["from_task"] == ("mrpc" if cycle % 2 else "qnli"),
                    f"{task}: invalid return history")
            ttft = row["ttft_ms"]
            require(all(math.isfinite(row[metric]) and row[metric] >= 0
                        for metric in TIMINGS if metric != "ttft_ms")
                    and row["switch_ms"] <= row["request_ms"], "Invalid timings")
            if ttft is None:
                require(row["stop_reason"] == "nonfinite_logits", "Missing first-token time")
            else:
                require(math.isfinite(ttft) and row["switch_ms"] <= ttft <= row["request_ms"], "Invalid TTFT")
            if method != "native" and cycle in AUDIT_CYCLES:
                require(isinstance(row["weight_match"], bool), f"{task}: missing weight check")
                if method in EXACT or published:
                    require(row["weight_match"] is (method in EXACT), f"{task}: sampled state differs")
            else:
                require(row["weight_match"] is None, f"{task}: unexpected weight check")
            token = first_difference(ref["tokens"], row["tokens"])
            if token is None and row["stop_reason"] == "nonfinite_logits":
                token = len(row["tokens"]) + 1
            if token is None:
                exact += 1
            if method in EXACT:
                require(token is None, f"{task}: merged path does not match resident outputs")
            elif method == "native":
                require(row["tokens"] == native_resident[row["id"]]["tokens"], f"{task}: Native changed its output")
            elif token is not None:
                divergences.append(dict(id=row["id"], block=block + 1, cycle=cycle,
                                        first_difference=token, quality=quality(task, row),
                                        canonical_quality=quality(task, ref)))
        values = {field: [r[field] for r in rows.values() if r[field] is not None] for field in TIMINGS}
        scores = [quality(task, r) for r in rows.values()]
        metrics[method, block] = dict(
            exact=exact, quality=sum(scores) if task == "gsm8k" else mean(scores),
            means={field: mean(times) if times else None for field, times in values.items()},
            p95={field: p95(times) if times else None for field, times in values.items()},
            **{field: stage[field] for field in ("correction_bytes", *MEMORY)},
        )
    if published:
        require(len({s["boot"] for s in stages.values()}) == 1, f"{task}: recorded boot grouping differs")

    pair_diffs, native_matched = [], []
    for block in (0, 1):
        shared, direct, native = (records[method, block] for method in ("shared", "direct", "native"))
        require(stages["shared", block]["boot"] == stages["direct", block]["boot"],
                f"{task}: Shared and Direct are on different boots")
        require(all(shared[key]["tokens"] == direct[key]["tokens"] for key in shared),
                f"{task}: Shared and Direct outputs differ")
        row = {field: mean(shared[key][field] - direct[key][field] for key in shared)
               for field in ("switch_ms", "ttft_ms", "request_ms")}
        row["p95_request_ms"] = (metrics["shared", block]["p95"]["request_ms"]
                                  - metrics["direct", block]["p95"]["request_ms"])
        pair_diffs.append(row)
        native_matched.extend((shared[k], native[k]) for k in shared if shared[k]["tokens"] == native[k]["tokens"])
    across_histories = sum(records["incremental", 0][k]["tokens"] != records["incremental", 1][k]["tokens"]
                           for k in reference)
    direct_build, shared_build = (data["build"][method] for method in ("direct", "shared"))
    require(direct_build["projection_records"] == 1512 and shared_build["projections"] == 756
            and not shared_build["old_edge_bank_read"] and not shared_build["old_edge_bank_generated"],
            f"{task}: standalone bank setup differs")
    for method, build in data["build"].items():
        require(all(build["payload_bytes"] == metrics[method, block]["correction_bytes"]
                    for block in (0, 1)), f"{task}: bank byte counts differ")
    if published:
        require(shared_build["mode_counts"] == {"target_low2": 756, "dense": 0},
                f"{task}: published modes differ")
    return dict(metrics=metrics, build=data["build"], divergences=divergences,
                pair_diffs=pair_diffs, across_histories=across_histories, native_matched=native_matched,
                native_distinct=sum(reference[k]["tokens"] != native_resident[k]["tokens"]
                                    for k in reference),
                telemetry=sum(stage["telemetry_samples"] for stage in stages.values()))


def write_table(path, layout, headers, rows):
    comment = ("% Paired from formal confirmation requests, with the original linear p95 convention."
               if path.stem == "paired_timing" else COMMENT)
    lines = [comment, r"\begin{tabular}{" + layout + "}", r"\toprule", *headers, r"\midrule"]
    lines.extend(row if isinstance(row, str) else " & ".join(map(str, row)) + r" \\" for row in rows)
    path.write_text("\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n")


def time_cell(value):
    return f"{value:.1f}" if value is not None else "--"


def tables_and_macros(audited, output):
    rows = {name: [] for name in FORMATS}
    observations = sum(entry["telemetry"] for entry in audited.values())
    macros = dict(TotalTimed=r"2{,}560", NVMLCount=f"{observations:,}".replace(",", "{,}"),
                  TestItems=128, PairedReturns=256)
    for task, result in audited.items():
        label, prefix = TASKS[task]
        metrics, build = result["metrics"], result["build"]
        rows["generation_main"].append(r"\multicolumn{8}{@{}l}{\textbf{%s}} \\" % label)
        rows["generation_by_block"].append(r"\multicolumn{9}{@{}l}{\textbf{%s}} \\" % label)
        for method, name in METHODS.items():
            left, right = (metrics[method, b] for b in (0, 1))
            scores = (f"{left['quality']} / {right['quality']}" if task == "gsm8k" else
                      f"{left['quality']:.4f} / {right['quality']:.4f}")
            available_ttft = [m["means"]["ttft_ms"] for m in (left, right) if m["means"]["ttft_ms"] is not None]
            ttft = mean(available_ttft) if available_ttft else None
            request = mean([left["means"]["request_ms"], right["means"]["request_ms"]]) / 1000
            require(left["correction_bytes"] == right["correction_bytes"], "Bank differs across blocks")
            rows["generation_main"].append((
                name, left["exact"], right["exact"], scores, time_cell(ttft), f"{request:.3f}",
                f"{gib(left['correction_bytes']):.2f}",
                f"{gib(max(left['peak_allocated_bytes'], right['peak_allocated_bytes'])):.2f}",
            ))
        for block in (0, 1):
            for index, (method, name) in enumerate(METHODS.items()):
                item = metrics[method, block]
                averages, tails = item["means"], item["p95"]
                rows["generation_by_block"].append((
                    str(block+1) if index == 0 else "", name, f"{averages['preceding_outbound_switch_ms']:.1f}",
                    f"{averages['switch_ms']:.1f}", time_cell(averages["ttft_ms"]), time_cell(tails["ttft_ms"]),
                    f"{averages['request_ms']/1000:.3f}", f"{tails['request_ms']/1000:.3f}",
                    f"{gib(item['peak_reserved_bytes']):.2f}",
                ))
            pair = result["pair_diffs"][block]
            timing = [rf"\({pair[field]:+.2f}\)" for field in ("switch_ms", "ttft_ms", "request_ms", "p95_request_ms")]
            rows["paired_timing"].append([label if block == 0 else "", str(block+1), *timing])
            if block == 0:
                rows["generation_by_block"].append(r"\addlinespace[2pt]")
        direct, shared = build["direct"], build["shared"]
        rows["generation_build"].append((
            label, f"{direct['encoding_wall_seconds']:.1f}", f"{direct['serialization_wall_seconds']:.1f}",
            f"{shared['session_setup_wall_seconds']:.1f}", f"{shared['calibration_wall_seconds']:.1f}",
            f"{shared['serialization_wall_seconds']:.1f}",
        ))
        if task == "gsm8k":
            rows["generation_main"].append(r"\midrule")
            rows["generation_by_block"].append(r"\midrule")
            rows["paired_timing"].append(r"\addlinespace[2pt]")

        matches, diverged = result["native_matched"], result["divergences"]
        direct_peak, shared_peak = (max(metrics[m, b]["peak_allocated_bytes"] for b in (0, 1))
                                    for m in ("direct", "shared"))
        host_ttft, shared_ttft = (mean(metrics[m, b]["means"]["ttft_ms"] for b in (0, 1))
                                  for m in ("host_copy", "shared"))
        host_e2e, shared_e2e = (mean(metrics[m, b]["means"]["request_ms"] for b in (0, 1))
                                for m in ("host_copy", "shared"))
        macros.update({prefix + key: value for key, value in dict(
            IncrementalDivergent=len(diverged), IncrementalDistinct=len({row["id"] for row in diverged}),
            IncrementalCrossHistory=result["across_histories"], NativeDistinct=result["native_distinct"],
            NativeMatched=len(matches),
            SharedMatchedRequest=f"{mean(x['request_ms'] for x, _ in matches)/1000:.3f}" if matches else "--",
            NativeMatchedRequest=f"{mean(x['request_ms'] for _, x in matches)/1000:.3f}" if matches else "--",
            BankSaving=percent_saved(direct["payload_bytes"], shared["payload_bytes"]),
            PeakSaving=percent_saved(direct_peak, shared_peak), HostTTFTSaving=percent_saved(host_ttft, shared_ttft),
            HostRequestSaving=percent_saved(host_e2e, shared_e2e),
        ).items()})
    output.mkdir(parents=True, exist_ok=True)
    for name, (layout, headers) in FORMATS.items():
        write_table(output / f"{name}.tex", layout, headers, rows[name])
    (output / "generation_macros.tex").write_text(
        COMMENT + "\n" + "".join(r"\newcommand{\%s}{%s}" % item + "\n" for item in macros.items()))
    return macros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=PUBLIC_EVIDENCE)
    parser.add_argument("--output", type=Path, default=Path("build/generation"))
    parser.add_argument("--with-figure", action="store_true", help="Render Figure 3 using Matplotlib")
    args = parser.parse_args()
    verify_release()
    published = args.evidence_dir.resolve() == PUBLIC_EVIDENCE
    audited = {task: audit_workload(args.evidence_dir / f"{task}.json.gz", task, published) for task in TASKS}
    output = args.output.resolve()
    macros = tables_and_macros(audited, output / "tables")
    switch_history_path = args.evidence_dir / "switch_history.json.gz"
    switch_history = None
    if switch_history_path.is_file():
        switch_history = rebuild_switch_history(switch_history_path, output / "tables/switch_history.tex")
    points = {task: result["divergences"] for task, result in audited.items()}
    if published:
        for expected in sorted((PUBLIC_EVIDENCE / "expected").glob("*.tex")):
            require((output / "tables" / expected.name).read_bytes() == expected.read_bytes(),
                    f"Reconstruction differs from the manuscript: {expected.name}")
        math_cases = [r for r in points["gsm8k"] if r["canonical_quality"] and not r["quality"]]
        require([(r["id"], r["cycle"], r["first_difference"]) for r in math_cases] ==
                [("test:1122", 39, 48), ("test:1122", 90, 48)], "GSM8K example differs")
        summary_cases = [r for r in points["dialogsum"] if r["id"] == "test_254"]
        require([(r["cycle"], r["first_difference"]) for r in summary_cases] == [(30, 16), (99, 31)],
                "DialogSum example differs")
        require([round(r["quality"] - r["canonical_quality"], 4) for r in summary_cases] == [-0.0370, 0.0275],
                "DialogSum example ROUGE-L changes differ")
    (output / "first_divergence.json").write_text(json.dumps(points, indent=2) + "\n")
    report = dict(
        evidence="paper" if published else "provided",
        requests=2560, resident_requests=512, distinct_inputs_per_task=128,
        table_files=list(FORMATS), summary=macros,
        paired_timing_ms={task: data["pair_diffs"] for task, data in audited.items()},
    )
    if switch_history is not None:
        report["table_files"].append("switch_history")
        report["switch_history"] = switch_history
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.with_figure:
        from plot_generation import render
        render(points, output / "figures/first_divergence.pdf")
    print(f"Audited 2,560 timed returns and 512 resident references. Outputs: {output}")


if __name__ == "__main__":
    main()
