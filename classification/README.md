# FTD: Target-Shared Corrections for Canonical LoRA Switching

FTD restores a designated merged FP16 LoRA checkpoint after a task switch.
An admitted low-rank update supplies a predictor. The correction bank stores
each target's low two raw-code bits once, the exact values at the union of
incoming exceptions, and a dense fallback when smaller. Predictor admission
remains source-specific.

## Reproduce the paper tables on a CPU

Use Python 3.12 in a clean environment. From this directory:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/check_codec.py
python scripts/reproduce.py --output build
```

The first script verifies the released encoder and CPU decoder on every
16-bit code, odd lengths, incoming exceptions, and dense fallback. The
second script verifies the release hashes and independently aggregates the
included recorded measurements. It writes LaTeX tables, a PDF/PNG figure,
a CSV, and JSON audit results to `build/`.

| Paper item | Generated file | Inputs |
| --- | --- | --- |
| Table I, five-method confirmation | `tables/confirmation50.tex` | 50 formal stages, 19,200 requests |
| Table II, offline Zstd | `tables/lossless_zstd.tex` | 3,780 measured coding units plus framing and reference bytes |
| Table III, switch frequency | `tables/frequency_mean.tex` | 54 formal stages, 20,736 requests |
| Figure 3, mean and p95 by frequency | `figures/switch_frequency.pdf` | Six stages per method and frequency |
| Table S7, complete same-boot blocks | `tables/reboot_sensitivity.tex` | Committed stages grouped by pseudonymous boot IDs |

The two serving experiments reuse a cohort of 384 questions. Repeated
stages provide timing observations. The p95 column averages the p95
computed separately for each stage. Figure 3 bars use the sample standard
deviation across stages. The audit compares candidate score vectors with the
archived Host-copy reference and checks the recorded prediction and cache
counts. It requires all the released formal stages and reports their GPU
observation counts.

## Execute the CUDA decoder

Use Linux with an NVIDIA GPU, a working CUDA-enabled PyTorch installation
and the GPU dependencies in `requirements-gpu.txt`. In the recorded GPU
environment, Python was 3.10, PyTorch was 2.10.0+cu128, CUDA was 12.8, Triton
was 3.6.0 and the driver was 580.173.02 on an RTX PRO 6000 Blackwell
Workstation Edition.

```bash
python -m pip install -r requirements-gpu.txt
python scripts/check_codec.py --cuda
```

This command exercises both the single-projection and grouped decode paths
using raw-code fixtures and the released Triton kernels. It needs no model
weights. Its output is a numerical smoke check.

## Rerun the Qwen3-8B measurements

The GPU runner uses the original backbone, the three trained adapters,
the exact serialized FP16 reference slices, and the frozen workload.
The required filenames and SHA-256 values are in `data/paper_assets.json`.
See [the asset and execution guide](docs/GPU_REPLAY.md) for the directory
layout, preflight, and complete commands.

The runner rebuilds corrections under the current runtime contract, admits
all canonical projections and grouped paths, and executes each stage in a
fresh process. The new run records its hardware and software contract and
compares its canonical scores with a locally executed Host-copy reference.
Original paper timings are available through the CPU reconstruction above.

The optional `scripts/measure_zstd.py` measures new Zstd frames from the
specified original or rebuilt banks and checks exact restoration. The asset
guide describes the inputs and command. The published Table II figures come
from the archived run reconstructed by `scripts/reproduce.py`.

## Files and coverage

- `runtime/` contains the measured target codec, grouped decoder, CUDA
  kernels, standalone Shared builder, and supporting FTD numerical and
  scoring implementation. Use the scripts above as public entry points.
- `data/` contains the publication projection of the formal request records,
  coding unit measurements, expected tables, runtime identity and asset pins.
  [Data provenance and schema](docs/DATA.md) explains the transformation.
- `data/runtime_provenance.json` records the source digests and released
  digests. The four measured Shared modules and standalone builder are
  byte-for-byte preserved. Relocated defaults in historical wrappers and
  their dependent integrity pins are recorded separately.
- `MANIFEST_SHA256.json` records the packaged files. Build outputs and local
  asset directories are ignored by Git.

This release reconstructs the main 8B tables, the switch-frequency figure
and the same-boot sensitivity table. The manuscript supplement records the
earlier Jetson and 0.6B protocols, policy diagnostics and distinct cohorts.
The portable GPU launcher is scoped to the main 8B protocol. The source
package is designed for local, serial, candidate-label scoring. See the
paper for the exact runtime, cache and memory accounting boundaries.

Publication terms for the FTD project sources and measurements are to be
set by their authors. The externally installed dependencies and separately
obtained model/data assets keep their respective terms.

