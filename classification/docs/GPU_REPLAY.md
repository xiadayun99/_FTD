# Main 8B GPU protocol

Use one NVIDIA GPU exposed to both CUDA and NVML on Linux. Before launching
a worker the runner waits for 30 seconds with zero observed compute
processes, at least 48 GiB free device memory and 104 GiB available host
memory. Allocate at least 64 GiB free space for the new output directory,
in addition to the external input assets. The logical experiment caps are
32 GiB tensor state and 512 MiB retained KV.

## Input assets

Supply three paths. The repository contains exact hashes for these inputs
in `data/paper_assets.json`.

| Argument | Required contents |
| --- | --- |
| `--model-dir` | The original Qwen3-8B local snapshot, including all five safetensors shards, the weight index, config and tokenizer files |
| `--reference-run` | `trained/lora/{mrpc,qnli,agnews}/adapter_config.json` and `adapter_model.safetensors`, plus `artifacts/references/metadata.json` and `{base,mrpc,qnli,agnews}.{json,safetensors}` |
| `--workload` | `engine/input/workload.json` from the completed 50-stage confirmation run |

The public backbone is Qwen3-8B. Check file identity with the digest pins
even if the directory name matches. The paper adapters and the serialized
FP16 reference slices are outputs from the authors' original runs. The
workload carries the original question text, effective tokens, labels,
heads and request order. Obtain the matching assets from the authors
subject to their distribution terms. Archived measurement records in this
repository are sufficient for the CPU reconstruction.

The three adapters use ranks 4, 16 and 8 for MRPC, QNLI and AG News,
respectively. The training protocol is retained in `runtime/vendor/runtime/disk_protocol.py`
and `disk_training.py`. It uses 3,000 rows per task, two epochs, a
microbatch of four, accumulation of four, a learning rate of
`2e-4` and 375 optimizer updates per task. The dataset filenames and
known training/exclusion hashes are in the asset specification. Exact
paper replay uses the original checkpoints and workload.

## Run

After installing a compatible CUDA stack and `requirements-gpu.txt`,
set the paths in this example to the local asset locations.

```bash
MODEL_DIR=/path/to/Qwen3-8B
REFERENCE_RUN=/path/to/original-8b-result
WORKLOAD=/path/to/fresh-confirmation/engine/input/workload.json

python scripts/run_gpu.py check-assets \
  --model-dir "$MODEL_DIR" --reference-run "$REFERENCE_RUN" --workload "$WORKLOAD"

python scripts/run_gpu.py run --experiment confirmation \
  --model-dir "$MODEL_DIR" --reference-run "$REFERENCE_RUN" --workload "$WORKLOAD" \
  --output runs/confirmation
```

Use `--experiment frequency --output runs/frequency` for the 54-stage
frequency sweep. `python scripts/run_gpu.py plan --experiment confirmation`
or `--experiment frequency` displays the scheduled jobs without model
loading.

Both runs verify hashes before touching CUDA. The runner builds Direct,
Shared and, for confirmation, Node banks in the run's `prepared/`
directory. It links to the reference and adapter inputs and leaves the
source files intact. Each worker performs the appropriate canonical
admission checks. Stage output is committed with SHA-256 hashes after a
successful exit. Timing includes the serialized request phases specified
by the paper. Correction construction and admission are outside request
timing.

Look for `status.json`, `summary.json`, `completed/` and `logs/` under
the output directory. To resume an interrupted run on the same boot, repeat
the command with `--resume`. Incomplete phase files are preserved separately.
After a reboot use a new directory so calibration and admission can be
performed under the current contract. Existing original measurement files
are external inputs.

`summary.json` records the new timings, accuracy counts, cache counts
and per-stage score comparison with the local Host-copy run. It also
records the local reference's score difference from the archived Host-copy
reference. The published hardware-specific numbers are audited separately
by the CPU workflow.

## Measure new Zstd frames

Install `libzstd` on Linux and the CPU dependencies. The paper measurement
used libzstd 1.5.5, level 3, 8 MiB independent blocks and three restoration
trials. For a run produced above:

```bash
python scripts/measure_zstd.py \
  --reference-run runs/confirmation/prepared --output runs/zstd
```

For the original bank files, pass the original 8B directory to
`--reference-run` and the original standalone Shared result to
`--shared-run`. Each frame is decoded and checked byte for byte.
Metadata checkpoints allow reusing measured units with identical inputs and
codec. The resulting sizes and times describe the supplied bytes and the
current machine. The audit from archived records reproduces the published
Table II.

