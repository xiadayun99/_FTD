# Measurement data and provenance

The release contains the fields needed to recompute the reported metrics.
The export checked the archived file digests, committed stage outputs, full
projection/path admission reports and GPU process observations against the
original experiment records. The archive digests and manifest counts are
listed in `data/provenance.json`.

- `reference.jsonl.gz` contains 384 Host-copy reference requests.
- `confirmation.jsonl.gz` has one line per committed formal stage. The row
  array holds the 384 timed requests and the stage record holds the measured
  memory fields and admission checks.
- `frequency.jsonl.gz` uses the same format. Its low, medium and high
  schedules preserve within-task order and use runs of 128, 8 and 1
  requests per task, yielding 2, 47 and 383 task switches per trace.
- `zstd_units.jsonl.gz` contains the 3,780 measurements of frame size,
  raw size, decoded-byte equality, frame counts and CPU timing.
  `zstd_framing.json` records the separately measured header/metadata sizes
  and the required immutable reference for XOR.
- `expected/` contains the printed LaTeX numerical tables used to check the
  reconstruction. Boot labels in Table S7 use release aliases.

`query` is the zero-based index in the original reference order and is
stable across both studies. Scores are keyed by candidate label. Timings
are in milliseconds and memory counters are bytes. `samples` contains
`[enumeration_verified, foreign_process_count]` for each formal worker
GPU sample. These samples cover the worker's observed admission and trace
execution. `boot` is a release-local alias for the actual boot reported in
the original evidence. Original prompt text, token sequences, process IDs,
host-specific paths and raw boot IDs are excluded from this release.

The serving reconstruction uses every released request. The grouping unit
for latency is a stage, with mean and linear-interpolated p95 computed
before averaging. The comparison counts unique inputs once for accuracy.
All five methods give 346 correct answers on the common 384-question
cohort. Shared, Direct, Node and Host copy match the archived canonical
candidate score vectors under the measured scorer contract.

The recorded measurements were made on the platform stated in the paper.
GPU observations establish the checks made by the sampler at its recorded
points. The published frequency aggregates exclude the recorded incomplete
attempts, including fourteen GPU-isolation failures and one earlier
SIGSEGV. Same-boot comparisons use only complete blocks for the applicable
methods. The full GPU run logs and original payloads are separate assets.

