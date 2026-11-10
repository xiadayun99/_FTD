# Target-Shared Corrections for Reproducible Generation under LoRA Switching

This is the reproduction package for the FTD manuscript. FTD restores a
designated merged FP16 LoRA checkpoint from an admitted low-rank predictor.
The target-shared bank stores one packed residue plane per target projection
and exact overrides for the union of incoming exceptions.

## Verify the recorded generation results

Run from the repository root with Python 3.10 or later. The table audit
uses only the standard library.

~~~bash
python scripts/reproduce.py --output build/generation
~~~

This audits 2,560 timed returns and 512 resident reference records for GSM8K
and DialogSum. It rebuilds Tables 1–4 and the numerical macros, checks them
against the manuscript's TeX files, and writes the first-divergence points
and an audit summary. Each task has 128 distinct test inputs and two switch
histories. The scripts count the histories separately for timing and sequence
comparisons. The same command audits the 1,280 paired depth records from
the fixed-peer replay and rebuilds the Appendix D table.

| Manuscript item | Output in build/generation |
| --- | --- |
| Table 1, sequence agreement, quality, timing and memory | tables/generation_main.tex |
| Table 2, matched Shared–Direct timing by block | tables/paired_timing.tex |
| Table 3, per-block distributions | tables/generation_by_block.tex |
| Table 4, offline bank construction | tables/generation_build.tex |
| Appendix D, fixed-peer switch-history replay | tables/switch_history.tex |
| Figure 3, first differing token | first_divergence.json and figures/first_divergence.pdf with plotting enabled |

For the PDF figure, codec vectors and the Appendix C classification audit,
use Python 3.12 with the CPU dependencies.

~~~bash
python -m pip install -r requirements-cpu.txt
python scripts/reproduce.py --output build/generation --with-figure
python classification/scripts/check_codec.py
python classification/scripts/reproduce.py --output build/classification
~~~

The codec check covers all 65,536 raw 16-bit codes, odd-length packing,
incoming exception unions and dense fallback. The classification audit
reconstructs the separate Qwen3-8B candidate-scoring, switch-frequency and
compression measurements summarized in Appendix C.

## Rerun on a GPU

The released GSM8K and DialogSum runners implement data freezing, adapter
training, independent FP16 merge, Direct/Shared construction, the GPU
admission checks and all five execution paths. The measured worker,
protocol and numerical operator files match the formal releases. The
runner's default paths and preflight entry point were adapted for a
repository checkout.

Use the original Qwen3-8B files and the two trained peer adapters with their
FP16 reference slices. The inputs are external to this repository and
checked by SHA-256. The hardware setup and exact commands are in the
[GPU replay guide](docs/GPU_REPLAY.md). Appendix C has a separate
[classification replay guide](classification/docs/GPU_REPLAY.md).
The Appendix D GPU entry point is [run_switch_history.sh](run_switch_history.sh).
It uses the original completed generation runs described in the
[switch-history replay notes](README_switch_history_zh.md).

## Repository layout

- [generation/gsm8k](generation/gsm8k) and [generation/dialogsum](generation/dialogsum) contain the measured generation workflows.
- [evidence](evidence) contains the publication projection of the generation measurements and the manuscript's numerical tables.
- [classification](classification) contains the independent Appendix C evidence, code and GPU replay entry point.
- [scripts](scripts) contains the CPU reconstruction and a raw-run evidence exporter.

See [Evidence and provenance](docs/EVIDENCE.md) for which quantities are
recomputed from the released records and which require a GPU run. The
datasets and model are obtained from their upstream sources under their
respective terms. The FTD source and recorded measurements carry no new
license grant in this bundle.
