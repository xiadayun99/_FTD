#!/usr/bin/env python3
"""Verify target-shared recovery with raw-code vectors. CUDA is optional."""
import argparse
import json

import numpy as np

from paper_protocol import ROOT
from shared_codec import decode_bits, encode_from_predictors, payload_bytes
from reproduce import verify_manifest


def fixtures():
    # Include every FP16 bit pattern, including patterns unsuitable for
    # arithmetic. These vectors are used as integers or copied by the decoder.
    all_codes = np.arange(65536, dtype=np.uint16)
    for name, target in (("all_codes", all_codes), ("odd_length", all_codes[:4099]),
                         ("exception_union", all_codes[:4099]), ("dense", all_codes[:1027])):
        raw = target.astype(np.int32)
        first = np.maximum(raw - 1, 0).astype(np.uint16)
        second = np.minimum(raw + 1, 65535).astype(np.uint16)
        if name == "exception_union":
            first[::97] ^= np.uint16(32768)
            second[4::89] ^= np.uint16(16384)
        if name == "dense":
            first ^= np.uint16(32768)
        yield name, target, (first, second)


def cpu_vectors():
    checked = []
    for name, target, predictions in fixtures():
        encoded = encode_from_predictors(target, predictions)
        expected_mode = "dense" if name == "dense" else "target_low2"
        if encoded["mode"] != expected_mode:
            raise ValueError(f"Wrong representation for {name}")
        if any(not np.array_equal(decode_bits(pred, encoded), target) for pred in predictions):
            raise ValueError(f"CPU recovery failed for {name}")
        if payload_bytes(encoded) > target.nbytes:
            raise ValueError(f"Payload exceeds dense fallback for {name}")
        if name == "exception_union":
            expected = np.flatnonzero(np.logical_or.reduce([
                np.abs(target.astype(np.int32) - pred.astype(np.int32)) > 1
                for pred in predictions
            ]))
            if not np.array_equal(encoded["indices"], expected):
                raise ValueError("Incoming exceptions were not unioned")
        checked.append((name, target, predictions, encoded))
    return checked


def cuda_vectors(checked):
    import sys
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and an NVIDIA GPU are required")
    vendor = ROOT / "runtime/vendor/runtime/vendor"
    sys.path[:0] = [str(vendor), str(vendor / "engine")]
    import rounding_patch
    from shared_codec import TargetPatch
    from shared_group import SharedPatchGroup

    def fp16_bits(raw):
        return torch.from_numpy(raw.copy().view(np.int16)).view(torch.float16)

    weights, patches, targets = {}, {}, {}
    for name, raw, predictions, encoded in checked:
        target = fp16_bits(raw)
        incoming = [fp16_bits(pred) for pred in predictions]
        patch = TargetPatch(
            tuple(target.shape), str(target.dtype), encoded["mode"],
            torch.from_numpy(encoded["packed"].copy()),
            torch.from_numpy(encoded["indices"].copy()),
            torch.from_numpy(encoded["values"].copy().view(np.int16)),
            {str(i): rounding_patch.digest(value) for i, value in enumerate(incoming)},
            rounding_patch.digest(target),
        ).to("cuda:0")
        for index, prediction in enumerate(incoming):
            value = prediction.to("cuda:0")
            patch.apply_(value, str(index), verify_input=True)
            torch.cuda.synchronize()
            if not rounding_patch.bit_equal(value, target):
                raise ValueError(f"CUDA decoder failed for {name} input {index}")
        weights[name], patches[name], targets[name] = incoming[0].to("cuda:0"), patch, target

    SharedPatchGroup(weights, patches).apply_()
    torch.cuda.synchronize()
    if any(not rounding_patch.bit_equal(weights[name], targets[name]) for name in weights):
        raise ValueError("Mixed grouped CUDA decoding failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also execute the released Triton decoder")
    args = parser.parse_args()
    verify_manifest(ROOT / "runtime", ROOT / "runtime/MANIFEST.json")
    checked = cpu_vectors()
    if args.cuda:
        cuda_vectors(checked)
    print(json.dumps({
        "cpu": "PASSED",
        "cuda": "PASSED" if args.cuda else "NOT_REQUESTED",
        "vectors": [
            dict(name=name, elements=target.size, mode=encoded["mode"], payload_bytes=payload_bytes(encoded))
            for name, target, _, encoded in checked
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

