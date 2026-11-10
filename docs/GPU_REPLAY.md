# GPU replay

## Environment

Use Linux with one NVIDIA GPU visible to both CUDA and NVML and Python
3.10 for the GPU environment. The formal results used an NVIDIA RTX PRO
6000 Blackwell Workstation Edition (96 GB), driver 580.173.02,
PyTorch 2.10.0+cu128, CUDA 12.8, Triton 3.6.0, Transformers 4.57.6 and
PEFT 0.19.1. The runner waits for 30 seconds of observed GPU idleness
and requires at least 48 GiB of free GPU memory and 104 GiB of available
host memory. Reserve at least 64 GiB of free disk space at the start
of each new generation run, in addition to the external inputs.

~~~bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
~~~

PyTorch publishes the CUDA 12.8 wheels in its
[installation reference](https://pytorch.org/get-started/previous-versions/).

## Required files

The same base and peer input files are used for both generation workloads.

| Argument | Required contents |
| --- | --- |
| model-dir | Original Qwen3-8B snapshot with five safetensors shards, index, configuration and tokenizer |
| reference-run | Original base, MRPC and QNLI FP16 mutable slices and metadata, plus MRPC and QNLI PEFT adapters |
| data-dir | Optional local folder with the pinned upstream JSONL files |

The original reference directory has this layout.

~~~text
reference_run/
  artifacts/references/
    metadata.json
    base.json
    base.safetensors
    mrpc.json
    mrpc.safetensors
    qnli.json
    qnli.safetensors
  trained/lora/
    mrpc/
      adapter_config.json
      adapter_model.safetensors
    qnli/
      adapter_config.json
      adapter_model.safetensors
~~~

Use the authors' original reference-run directory or obtain its matching
files separately. The repository contains the exact filename and SHA-256
pins in [paper_assets.json](../generation/gsm8k/paper_assets.json).
The public backbone is [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
Verify its snapshot against the pins. These are the original trained peer
adapters and FP16 reference artifacts. The target adapters for GSM8K and
DialogSum are trained during their respective runs. Each runner constructs
new Direct and Shared banks under the local arithmetic contract.

The data phase verifies the upstream files before freezing the prompts.

- GSM8K uses the official main split files train.jsonl and test.jsonl. [OpenAI repository](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)
- DialogSum uses dialogsum.train.jsonl, dialogsum.dev.jsonl and dialogsum.test.jsonl at revision 848cc8c0. [Dataset revision](https://github.com/cylnlp/dialogsum/tree/848cc8c05e0a04def79326539de827f1b28786a0/DialogSum_Data)

## Run each workload

Run the following from the repository root. The CPU-only preflight verifies
the input files before reserving GPU resources. The two work directories
must be separate from the input directories.

~~~bash
MODEL_DIR=/path/to/Qwen3-8B
REFERENCE_RUN=/path/to/original-8b-result

for task in gsm8k dialogsum; do
  python "generation/$task/run_generation.py" check-assets \
    --model-dir "$MODEL_DIR" --reference-run "$REFERENCE_RUN" || break
done
~~~

Set TASK to gsm8k or dialogsum and run the pilot. For an offline host, append
--data-dir /path/to/pinned-jsonl-files to each invocation.

~~~bash
TASK=gsm8k
RUN_DIR="runs/$TASK"

python "generation/$TASK/run_generation.py" plan
python -u "generation/$TASK/run_generation.py" pilot \
  --model-dir "$MODEL_DIR" --reference-run "$REFERENCE_RUN" --output "$RUN_DIR"
~~~

Inspect status.json and pilot_summary.json under the output directory.
After the pilot reports COMPLETED, use the same paths for confirmation.

~~~bash
python -u "generation/$TASK/run_generation.py" confirm \
  --model-dir "$MODEL_DIR" --reference-run "$REFERENCE_RUN" --output "$RUN_DIR"
~~~

Change TASK and RUN_DIR to rerun the other workload. Repeat an interrupted
command with the same paths and release to resume its verified stages.
Use a fresh output directory if the input files, software or hardware
contract changes. Each confirmation emits confirm_summary.json, per-stage
reports and requests.jsonl files. GPU admission and preparation precede the
timed return switch. Per-request timing excludes the preceding outbound
switch. The controller records and checks process observations, stage
outputs and the arithmetic contract.

For a completed run under the same protocol, export and audit its evidence.

~~~bash
python scripts/export_generation.py \
  --gsm8k-run runs/gsm8k --dialogsum-run runs/dialogsum --output build/replay-evidence
python scripts/reproduce.py --evidence-dir build/replay-evidence --output build/replay-audit
~~~

The paper data audit checks the published numerical tables against the
included archived measurements. A new hardware run produces its own
latency observations and is evaluated against its local resident reference.
The recorded GPU identifiers and original experiment logs are excluded from
the repository.
