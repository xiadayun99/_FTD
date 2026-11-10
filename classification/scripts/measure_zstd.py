#!/usr/bin/env python3
"""Measure the paper's block Zstd comparison on supplied FP16 references and banks."""
import argparse
import fcntl
from pathlib import Path

from paper_protocol import ROOT, TASKS
from reproduce import read, verify_manifest
from zstd_baseline import (
    MIB, TensorFile, Zstd, append_checkpoint, canonical, digest_bytes, dump,
    load_checkpoints, measure_unit, sha, summarize,
)


def make_plans(source, shared):
    inputs = {}
    plans = []

    def verified_tensor(path, expected):
        if sha(path, progress=True) != expected:
            raise ValueError(f"Tensor input digest differs: {path.name}")
        inputs[str(path)] = expected
        return TensorFile(path)

    def add(group, label, tensor_file, units, reference=None, metadata=None):
        header = dict(
            format="ftd_block_zstd_v1",
            transform="xor_immutable_base" if reference else "identity",
            original_safetensors_header=tensor_file.header_bytes.decode("utf-8"),
            tensor_order=[name for _, names in units for name in names],
            source_file_sha256=inputs[str(tensor_file.path)],
            reference_file_sha256=inputs[str(reference.path)] if reference else None,
        )
        plans.append(dict(group=group, label=label, file=tensor_file, units=units,
                          reference=reference, metadata=metadata, header=header))

    reference_dir = source / "artifacts/references"
    base = verified_tensor(reference_dir / "base.safetensors", read(reference_dir / "base.json")["file_sha256"])
    if len(base.tensors) != 252 or any(value["dtype"] != "F16" for value in base.tensors.values()):
        raise ValueError("Expected the paper's 252 FP16 base projections")
    for task in TASKS:
        target = verified_tensor(reference_dir / f"{task}.safetensors",
                                 read(reference_dir / f"{task}.json")["file_sha256"])
        if set(target.tensors) != set(base.tensors):
            raise ValueError("Target/base projection coverage differs")
        units = [(name, [name]) for name in sorted(target.tensors)]
        add("canonical_mutable_raw_zstd", task, target, units)
        add("canonical_mutable_xor_base_zstd", task, target, units, reference=base)

    for group, directory in (
        ("direct_patch_zstd", source / "artifacts/patches/direct"),
        ("shared_patch_zstd", shared / "artifacts/shared_target"),
    ):
        metadata = directory / "patch_metadata.json"
        inputs[str(metadata)] = sha(metadata)
        index = read(metadata)
        payload = verified_tensor(directory / "patches.safetensors", index["payload_sha256"])
        units = []
        covered = []
        for kind, owners in sorted(index["banks"].items()):
            for owner, entries in sorted(owners.items()):
                for name, entry in sorted(entries.items()):
                    fields = [entry["key"] + "." + field for field in ("packed", "indices", "values")]
                    size = sum(payload.tensors[field]["data_offsets"][1] -
                               payload.tensors[field]["data_offsets"][0] for field in fields)
                    if size != entry["metadata"]["bytes"]:
                        raise ValueError(f"Patch payload accounting differs for {name}")
                    units.append((f"{kind}/{owner}/{name}", fields))
                    covered.extend(fields)
        if len(set(covered)) != len(covered) or set(covered) != set(payload.tensors):
            raise ValueError("Patch tensor coverage differs")
        if payload.payload_bytes != index["tensor_bytes"]:
            raise ValueError("Patch tensor size differs")
        add(group, "bank", payload, units, metadata=metadata)
    return plans, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="directory containing artifacts/references and artifacts/patches/direct")
    parser.add_argument("--shared-run", type=Path, help="directory containing artifacts/shared_target")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_manifest(ROOT, ROOT / "MANIFEST_SHA256.json")
    source = args.reference_run.resolve()
    shared = args.shared_run.resolve() if args.shared_run else source
    out = args.output.resolve()
    if out.is_relative_to(source) or out.is_relative_to(shared):
        raise ValueError("Place new results outside the input directories")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        codec = Zstd(level=3)
        plans, inputs = make_plans(source, shared)
        config = dict(level=3, block_bytes=8 * MIB, decode_repeats=3,
                      zstd_version=codec.version, inputs_sha256=inputs,
                      implementation_sha256=sha(ROOT / "runtime/zstd_baseline.py"),
                      runner_sha256=sha(Path(__file__)))
        identity = digest_bytes(canonical(config))
        protocol = out / "protocol.json"
        if protocol.exists() and read(protocol) != config:
            raise ValueError("An existing measurement uses different inputs or a different codec")
        dump(protocol, config)
        checkpoint = out / "completed_units.jsonl"
        done = load_checkpoints(checkpoint, identity)
        for plan in plans:
            for unit, names in plan["units"]:
                key = plan["group"] + "/" + plan["label"] + "/" + unit
                if key in done:
                    continue
                measurement = measure_unit(codec, plan["file"], names, plan["reference"],
                                           config["block_bytes"], config["decode_repeats"])
                record = dict(key=key, measurement=measurement, protocol_sha256=identity)
                append_checkpoint(checkpoint, record)
                done[key] = record
            print(f"Verified {plan['group']} / {plan['label']}", flush=True)
        for path, expected in inputs.items():
            if sha(path) != expected:
                raise ValueError(f"Input changed during measurement: {path}")
        dump(out / "summary.json", summarize(plans, done, config))
    print(out / "summary.json")


if __name__ == "__main__":
    main()

