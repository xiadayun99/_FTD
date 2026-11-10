"""Export the measured generation fields without prompt, answer text or machine identifiers."""

from __future__ import annotations

import argparse
import gzip
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from statistics import mean
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "dialogsum")
TIMES = ("preceding_outbound_switch_ms", "switch_ms", "ttft_ms", "request_ms")
MEMORY = ("peak_allocated_bytes", "peak_reserved_bytes")
STAGE_FIELDS = ("correction_bytes", "telemetry_samples", *MEMORY)
RUNTIME_FIELDS = ("torch", "cuda", "triton", "transformers", "peft", "gpu", "driver")
BUILD_FIELDS = {
    "direct": (
        "payload_bytes", "projection_records", "encoding_wall_seconds",
        "serialization_wall_seconds",
    ),
    "shared": (
        "payload_bytes", "projections", "session_setup_wall_seconds",
        "calibration_wall_seconds", "serialization_wall_seconds", "mode_counts",
        "old_edge_bank_read", "old_edge_bank_generated",
    ),
}


def load_protocol(task):
    path = ROOT / "generation" / task / "generation_protocol.py"
    spec = spec_from_file_location(f"ftd_{task}_protocol", path)
    protocol = module_from_spec(spec)
    spec.loader.exec_module(protocol)
    return protocol


def selected(source, fields):
    return {field: source[field] for field in fields}


def export_record(protocol, task, row, codes, digest):
    # One stable codebook per input preserves equality and positions.
    # The original token IDs and codebook are excluded from the export.
    tokens = [codes.setdefault(token, len(codes)) for token in row["generated_ids"]]
    public = dict(
        id=row["id"], cycle=row["cycle"], from_task=row["from_task"],
        tokens=tokens, stop_reason=row["stop_reason"],
        weight_match=None if row["weight_digest"] is None else row["weight_digest"] == digest,
        **selected(row, (*TIMES, *MEMORY)),
    )
    if task == "gsm8k":
        public.update(prediction=row["prediction"], gold=row["gold"])
    else:
        words = protocol.ROUGE_TOKEN.findall(row["text"].lower())
        ref_words = [protocol.ROUGE_TOKEN.findall(ref.lower()) for ref in row["references"]]
        counts = [[protocol.lcs_length(words, ref), len(words) + len(ref)] for ref in ref_words]
        score = 0.0 if row["stop_reason"] == "nonfinite_logits" else mean(
            2 * overlap / length for overlap, length in counts)
        if score != row["rouge_l_f1"]:
            raise ValueError(f"DialogSum score export differs for {row['id']}")
        public["rouge_l_counts"] = counts
    return public


def export(task, run, output):
    protocol = load_protocol(task)
    saved = protocol.read_json(run / "confirm_summary.json")
    with patch.object(protocol, "write_json"):
        audited = protocol.summarize(run, "confirm")
    if audited != saved or audited["status"] != "COMPLETED":
        raise ValueError(f"{task}: original confirmation summary does not pass the release audit")
    codebooks, boot_aliases = {}, {}
    stages = []
    for job in protocol.schedule("confirm"):
        directory = run / "stages" / job["key"]
        committed = protocol.read_json(run / "completed" / f"{job['key']}.json")
        if committed["job"] != job or not committed["outputs"]:
            raise ValueError(f"Invalid stage commitment: {job['key']}")
        for name, digest in committed["outputs"].items():
            path = run / name
            if not path.resolve().is_relative_to(directory.resolve()) or protocol.sha(path) != digest:
                raise ValueError(f"Invalid committed output: {name}")
        records = protocol.rows(directory / "requests.jsonl")
        report = protocol.read_json(directory / "report.json")
        digest = report["canonical_gsm8k_digest" if task == "gsm8k" else "canonical_target_digest"]
        boot = report["boot_id"]
        alias = boot_aliases.setdefault(boot, f"boot-{len(boot_aliases) + 1}")
        public_records = [
            export_record(protocol, task, row, codebooks.setdefault(row["id"], {}), digest)
            for row in records
        ]
        stages.append(dict(
            method=job["method"], block=None if job["resident"] else job["block"],
            boot=alias, rows=public_records, **selected(report, STAGE_FIELDS),
        ))
    direct = protocol.read_json(run / "artifacts/patches/direct/build_report.json")
    shared = protocol.read_json(run / "artifacts/shared_target/build_report.json")
    evidence = dict(
        schema=1, task=task, protocol=protocol.PROTOCOL_VERSION,
        runtime=selected(protocol.read_json(run / "runtime_contract.json"), RUNTIME_FIELDS),
        data_sources=protocol.read_json(run / "inputs/cohort.json")["source_sha256"],
        build=dict(direct=selected(direct, BUILD_FIELDS["direct"]),
                   shared=selected(shared, BUILD_FIELDS["shared"])),
        stages=stages,
    )
    out = output / f"{task}.json.gz"
    payload = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    out.write_bytes(gzip.compress(payload.encode(), mtime=0))
    print(f"{task}: {len(stages)} audited stages, {sum(len(s['rows']) for s in stages)} records -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-run", type=Path, required=True)
    parser.add_argument("--dialogsum-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        export(task, getattr(args, f"{task}_run"), args.output)


if __name__ == "__main__":
    main()
