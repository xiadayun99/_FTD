"""Rebuild Appendix D from the processed, paired switch-history records."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
TASKS = ("gsm8k", "dialogsum")
DEPTHS = (0, 2, 8, 32, 128)
INPUTS = 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(evidence_path):
    with gzip.open(evidence_path, "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    require(evidence["schema"] == 1 and set(evidence["tasks"]) == set(TASKS),
            "Unexpected switch-history evidence schema")
    audited = {}
    for task in TASKS:
        data = evidence["tasks"][task]
        reference = data["reference"]
        ids = [row["id"] for row in reference]
        require(len(ids) == len(set(ids)) == INPUTS, f"{task}: invalid reference cohort")
        groups = data["groups"]
        require([group["return_cycles"] for group in groups] == list(DEPTHS),
                f"{task}: switch depths differ")
        per_depth, ever_different = [], set()
        for group in groups:
            depth, rows = group["return_cycles"], group["rows"]
            require(group["final_source"] == ("qnli" if depth else None),
                    f"{task}: final source differs at depth {depth}")
            require([row["id"] for row in rows] == ids,
                    f"{task}: unpaired cohort at depth {depth}")
            different = {row["id"] for ref, row in zip(reference, rows)
                         if ref["sequence_codes"] != row["sequence_codes"]}
            require(depth > 0 or not different, f"{task}: resident sequences differ")
            ever_different.update(different)
            result = dict(return_cycles=depth, sequence_mismatches=len(different))
            if task == "gsm8k":
                result["correct"] = sum(row["prediction"] == ref["gold"]
                                        for ref, row in zip(reference, rows))
            else:
                scores = [row["rouge_l_f1"] for row in rows]
                require(all(0.0 <= score <= 1.0 for score in scores), "ROUGE-L F1 out of range")
                result["mean_rouge_l_f1"] = mean(scores)
            per_depth.append(result)
        audited[task] = dict(distinct_inputs=INPUTS, distinct_inputs_ever_different=len(ever_different),
                             per_depth=per_depth)
    return audited


def rebuild(evidence_path, table_path):
    audited = audit(evidence_path)
    lines = [
        "% Recomputed from the fixed-peer GPU replay records.",
        r"\begin{tabular}{@{}rcccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{DialogSum} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Cycles & Seq.\ diff. & Correct & Seq.\ diff. & ROUGE-L F1 \\",
        r"\midrule",
    ]
    for depth, gsm, dialog in zip(DEPTHS, *(audited[task]["per_depth"] for task in TASKS)):
        row = (f"{depth:<3} & {gsm['sequence_mismatches']:<2} & {gsm['correct']} & "
               f"{dialog['sequence_mismatches']} & {dialog['mean_rouge_l_f1']:.4f}")
        lines.append(row + r" \\")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n",
                          encoding="utf-8")
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/switch_history.tex"))
    args = parser.parse_args()
    rebuild(EVIDENCE / "switch_history.json.gz", args.output)
    require(args.output.read_bytes() == (EVIDENCE / "expected/switch_history.tex").read_bytes(),
            "Reconstruction differs from Appendix D")
    print(f"Audited 1,280 paired depth records. Table: {args.output}")


if __name__ == "__main__":
    main()
