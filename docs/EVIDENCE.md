# Evidence and provenance

The generation export contains 12 formal stages per task. Ten timed stages
cover five methods on the same 128 held-out inputs in two reverse-order
blocks. Two resident stages supply the merged and native references.
The exporter verifies the completed stage-file hashes and reruns the
original protocol's CPU audit before projecting the records.

Each public row contains a benchmark ID, return cycle, incoming peer,
timings, allocator peaks, stop reason and the observed bit-check outcome
where the original trace sampled it. Token IDs are remapped separately
for each input across all methods and both histories. This preserves exact
sequence equality, lengths and first differing positions without exposing
the original vocabulary IDs or generated text. Boot labels are local
aliases. PIDs, GPU UUIDs, host paths and raw benchmark text are excluded.

For GSM8K, parsed numeric predictions and numeric gold values allow
the audit to recompute answer exact match. For DialogSum, each of the
three reference pairs contains the longest-common-subsequence count
and the combined candidate/reference word count. The audit computes
the mean of 2 times the LCS count divided by the combined count for
each dialogue, then averages across dialogues. The exporter computed
these counts using the recorded lowercase ASCII alphanumeric tokenization.
The published evidence supports the score arithmetic. Recomputing token
decoding and LCS directly from raw text is part of the GPU protocol and its
original audit.

TTFT and end-to-end are measured from the start of the return switch.
A p95 is linear interpolation within a block. Table 2's last column
is the difference between the two method p95s in that block.
The two blocks repeat the same inputs. The retained bank counts are bytes
of repair tensors, and peak allocated and reserved bytes refer to the
PyTorch allocator. The preparation table uses the archived build records.

The measured generation worker, protocol, and numerical runtime were
retained byte for byte. The public controller changes its default paths
and adds a preflight command. The prior and public release digests are
recorded in [provenance.json](../evidence/provenance.json). The repository
manifest pins the distributed files.

Appendix C uses the separate [classification records](../classification/docs/DATA.md).
It has a different task set and uses version-matched prefix KV reuse.
Its measurements are reported independently of the two generation runs.

Appendix D uses the [fixed-peer replay records](../evidence/switch_history.json.gz).
Each task contributes the same 128 frozen inputs at each of five depths.
Sequence codes preserve exact equality for an input across depths and its
resident reference. The projection retains parsed GSM8K predictions and
gold answers, and the measured per-input DialogSum ROUGE-L F1. The CPU
audit recomputes sequence mismatch counts, answer exact match, and mean
ROUGE-L F1 from those fields. The GPU entry point uses the original
completed generation runs to verify the resident and depth anchors.
