"""Frozen DialogSum cohorts, return schedule, and CPU audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
import time
import unicodedata
from urllib.request import Request, urlopen

TASK = "dialogsum"
PEERS = ("mrpc", "qnli")
TASKS = (*PEERS, TASK)
METHODS = ("host_copy", "incremental", "direct", "shared", "native")
EXACT = ("host_copy", "direct", "shared")
RANK = 8
ALPHA = 16
TRAIN_COUNT = 768
DEV_COUNT = 8
WARMUP_COUNT = 2
TEST_COUNT = 128
EPOCHS = 1
ACCUMULATION = 8
LEARNING_RATE = 2e-4
SEED = 20261109
MAX_CONTEXT = 1024
MAX_NEW_TOKENS = 256
PILOT_AGES = (1, 2, 4, 8, 12, 16, 24, 32)
AUDIT_AGES = frozenset((1, 2, 8, 32, 64, 128))
GIB = 2**30
LIVE_CAP = 32 * GIB
MIN_GPU_FREE = 48 * GIB
MIN_HOST_AVAILABLE = 104 * GIB
START_DISK = 64 * GIB
PROTOCOL_VERSION = "ftd_acl2027_dialogsum_return_v1"
ARCHIVE_PREFIX = "ftd_acl_dialogsum_"
DATA_REVISION = "848cc8c05e0a04def79326539de827f1b28786a0"
DATA_URL = f"https://raw.githubusercontent.com/cylnlp/dialogsum/{DATA_REVISION}/DialogSum_Data"
DATA = {
    "train": (12460, 12068822, "660238631152c062e67186da6622c1e8925579e82563d66d611abb9dcc19bb72"),
    "dev": (500, 471830, "436721e0a3571eef7072964f80e042fb151d8c6178db1a688ce8f803bbf0a13f"),
    "test": (500, 639707, "6de36eca7e7b9b10975fd5ea3f47391df172d8b3c15d03d84d763cbaab015fda"),
}
INSTRUCTION = (
    "Summarize the following dialogue in concise English from an observer's perspective. "
    "Return only the summary.\n\n"
)
ROUGE_TOKEN = re.compile(r"[a-z0-9]+")
QUALITY_METRIC = "mean_reference_rouge_l_f1_ascii_words_no_stem"


def read(path):
    return Path(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    partial.replace(path)


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def append_row(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def lcs_length(left, right):
    """Longest common subsequence in one row of dynamic-programming storage."""
    if len(left) < len(right):
        left, right = right, left
    lengths = [0] * (len(right) + 1)
    for left_token in left:
        diagonal = 0
        for index, right_token in enumerate(right, 1):
            above = lengths[index]
            lengths[index] = diagonal + 1 if left_token == right_token else max(above, lengths[index-1])
            diagonal = above
    return lengths[-1]


def summary_quality(text, references, stop_reason):
    """Mean per-reference ROUGE-L F1, plus a flag for an empty decoded summary."""
    words = ROUGE_TOKEN.findall(text.lower())
    if not words or stop_reason == "nonfinite_logits":
        return 0.0, not words
    scores = []
    for reference in references:
        ref_words = ROUGE_TOKEN.findall(reference.lower())
        scores.append(2 * lcs_length(words, ref_words) / (len(words) + len(ref_words)))
    return mean(scores), False


def dialogue_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def indexed(split, examples):
    reference_fields = ("summary1", "summary2", "summary3") if split == "test" else ("summary",)
    result = []
    for example in examples:
        item_id = example["fname"]
        dialogue = example["dialogue"].strip()
        references = [example[field].strip() for field in reference_fields]
        if (not item_id.startswith(f"{split}_") or not dialogue or
                not all(ROUGE_TOKEN.search(reference.lower()) for reference in references)):
            raise ValueError(f"Invalid DialogSum row: {split}/{item_id}")
        result.append(dict(id=item_id, dialogue=dialogue, dialogue_sha256=dialogue_key(dialogue),
                           references=references))
    if len({row["id"] for row in result}) != len(result):
        raise ValueError(f"Duplicate DialogSum IDs in {split}")
    return result


def unique_order(examples, seed, excluded):
    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    seen = set(excluded)
    result = []
    for row in ordered:
        key = row["dialogue_sha256"]
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def freeze_cohorts(tokenizer, sources, train_count=TRAIN_COUNT, test_count=TEST_COUNT,
                   dev_count=DEV_COUNT, warmup_count=WARMUP_COUNT):
    """Select in seeded order after split-overlap, duplicate, and length checks."""
    if min(train_count, test_count, dev_count, warmup_count) < 1:
        raise ValueError("Every DialogSum cohort needs a positive quota")
    splits = {split: indexed(split, sources[split]) for split in DATA}
    keys = {split: {row["dialogue_sha256"] for row in examples} for split, examples in splits.items()}
    counts = {"train": train_count, "dev": dev_count+warmup_count, "test": test_count}
    selected, filtering = {}, {}
    for offset, (split, examples) in enumerate(splits.items()):
        other = set.union(*(values for name, values in keys.items() if name != split))
        order = unique_order(examples, SEED+offset, other)
        chosen = []
        overlength = 0
        for row in order:
            tokens = prompt_ids(tokenizer, row["dialogue"])
            item = dict(id=row["id"], dialogue_sha256=row["dialogue_sha256"])
            if split == "train":
                tokens, labels = encode_training(tokenizer, tokens, row["references"][0])
                fits = len(tokens) <= MAX_CONTEXT
                item.update(tokens=tokens, labels=labels)
            else:
                fits = len(tokens) + MAX_NEW_TOKENS <= MAX_CONTEXT
                item.update(prompt_ids=tokens, prompt_sha256=identity(tokens), references=row["references"])
            if not fits:
                overlength += 1
                continue
            chosen.append(item)
            if len(chosen) == counts[split]:
                break
        if len(chosen) != counts[split]:
            raise ValueError(f"Too few eligible DialogSum rows in {split}: {len(chosen)}")
        selected[split] = chosen
        filtering[split] = dict(source_rows=len(examples), distinct_dialogues=len(keys[split]),
            excluded_cross_split_dialogues=len(keys[split] & other), unique_candidates=len(order),
            overlength_skipped_before_quota=overlength)
    workload = dict(dev=selected["dev"][:dev_count], warmup=selected["dev"][dev_count:],
                    test=selected["test"])
    return selected["train"], workload, filtering


def official_file(root, split, data_dir=None):
    nrows, size, expected = DATA[split]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"dialogsum.{split}.jsonl"

    def check(candidate):
        if candidate.stat().st_size != size or sha(candidate) != expected:
            raise ValueError(f"Expected the pinned official DialogSum {path.name}: {candidate}")

    if not path.exists():
        if data_dir is not None:
            source = Path(data_dir) / path.name
            check(source)
            data = source.read_bytes()
        else:
            url = f"{DATA_URL}/{path.name}"
            error = None
            for attempt in range(3):
                try:
                    request = Request(url, headers={"User-Agent": "FTD-DialogSum-research"})
                    with urlopen(request, timeout=45) as response:
                        data = response.read(size + 1)
                    if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f"Unexpected content at {url}")
                    error = None
                    break
                except OSError as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(2**attempt)
            if error is not None:
                raise OSError(
                    f"Could not download {url}. Put the three pinned DialogSum JSONL files "
                    "in a local directory and retry with --data-dir."
                ) from error
        tmp = path.with_suffix(".partial")
        tmp.write_bytes(data)
        check(tmp)
        tmp.replace(path)
    check(path)
    examples = rows(path)
    if len(examples) != nrows:
        raise ValueError(f"Wrong number of {split} examples")
    return examples


def prompt_ids(tokenizer, dialogue):
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + dialogue}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    if not isinstance(result, list) or not result or any(type(token) is not int for token in result):
        raise TypeError("Expected one Qwen3 chat-template token sequence")
    return result


def encode_training(tokenizer, prompt, summary):
    target = tokenizer.encode(summary, add_special_tokens=False)
    if not target or tokenizer.eos_token_id is None:
        raise ValueError("Expected a nonempty completion and a Qwen3 EOS token")
    target = target + [tokenizer.eos_token_id]
    return prompt + target, [-100] * len(prompt) + target


def prepare_data(root, model_dir, data_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False, local_files_only=True)
    sources = {split: official_file(root / "dataset_source", split, data_dir) for split in DATA}
    training_plan, workload, filtering = freeze_cohorts(tokenizer, sources)
    cohorts = dict(train=training_plan, **workload)
    keys = [row["dialogue_sha256"] for group in cohorts.values() for row in group]
    if len(set(keys)) != len(keys):
        raise ValueError("Frozen DialogSum cohorts contain overlapping dialogues")
    write_json(root / "inputs/training.json", training_plan)
    write_json(root / "inputs/workload.json", workload)
    write_json(root / "inputs/cohort.json", dict(
        protocol=PROTOCOL_VERSION, source_url=DATA_URL, source_revision=DATA_REVISION,
        source_sha256={key: meta[2] for key, meta in DATA.items()},
        split_seeds={split: SEED+index for index, split in enumerate(DATA)},
        distinct_dialogue_counts={
            key: len(value) for key, value in cohorts.items()},
        normalized_dialogue_overlap=0, official_test_seen_by_sft=False, filtering=filtering,
        test_references_per_dialogue=3, quality_metric=QUALITY_METRIC,
        max_training_tokens=max(len(row["tokens"]) for row in training_plan),
        max_evaluation_prompt_tokens=max(len(row["prompt_ids"]) for key in workload for row in workload[key]),
        training_sha256=sha(root / "inputs/training.json"),
        workload_sha256=sha(root / "inputs/workload.json"),
    ))
    print(f"DATA_FROZEN: train={len(training_plan)} dev={len(workload['dev'])} "
          f"warmup={len(workload['warmup'])} test={len(workload['test'])}", flush=True)


def peer(cycle):
    if cycle < 1:
        raise ValueError("Cycle is one-based")
    return PEERS[(cycle - 1) % len(PEERS)]


def return_cases(workload, scope, block):
    if scope == "pilot" and block == 0:
        examples, cycles = workload["dev"], PILOT_AGES
    elif scope == "confirm" and block in (0, 1):
        examples = workload["test"] if block == 0 else workload["test"][::-1]
        cycles = tuple(range(1, len(examples) + 1))
    else:
        raise ValueError("Unexpected workload scope or block")
    if len(examples) != len(cycles):
        raise ValueError("Return workload and cycle counts differ")
    return [(item, cycle) for item, cycle in zip(examples, cycles)]


def schedule(scope):
    if scope not in ("pilot", "confirm"):
        raise ValueError("Unexpected experiment scope")
    plan = [
        dict(key=f"{scope}_resident_{method}", phase="trace", scope=scope, block=0,
             method=method, resident=True)
        for method in ("host_copy", "native")
    ]
    orders = [METHODS] if scope == "pilot" else [
        ("shared", "direct", "native", "incremental", "host_copy"),
        ("host_copy", "incremental", "native", "direct", "shared"),
    ]
    for block, order in enumerate(orders):
        for method in order:
            plan.append(dict(key=f"{scope}_r{block}_{method}", phase="trace", scope=scope,
                             block=block, method=method, resident=False))
    return plan


def first_difference(left, right):
    for offset, (a, b) in enumerate(zip(left, right), 1):
        if a != b:
            return offset
    if len(left) != len(right):
        return min(len(left), len(right)) + 1
    return None


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metrics(values):
    return dict(mean=mean(values), median=percentile(values, .5), p95=percentile(values, .95))


def audit_stage(job, workload, records, report):
    expected = return_cases(workload, job["scope"], job["block"])
    if job["resident"]:
        expected = [(item, 0) for item, _ in expected]
    if len(expected) != len(records) or report["requests"] != len(expected):
        raise ValueError(f"Incomplete trace: {job['key']}")
    if report["job"] != job or not report["gpu_isolation_verified"] or not report["boot_id"]:
        raise ValueError(f"Unexpected or contaminated stage report: {job['key']}")
    for position, ((case, cycle), row) in enumerate(zip(expected, records)):
        fields = dict(position=position, id=case["id"], prompt_sha256=case["prompt_sha256"],
                      references=case["references"], cycle=cycle, completed_switches=2 * cycle,
                      from_task=peer(cycle) if cycle else TASK)
        if any(row[key] != value for key, value in fields.items()):
            raise ValueError(f"Frozen request changed: {job['key']} position {position}")
        tokens = row["generated_ids"]
        if row["stop_reason"] == "nonfinite_logits":
            if job["method"] != "incremental" or len(tokens) >= MAX_NEW_TOKENS:
                raise ValueError("Unexpected nonfinite decoder status")
        elif not 1 <= len(tokens) <= MAX_NEW_TOKENS or row["stop_reason"] not in ("eos", "length_limit"):
            raise ValueError("Unexpected generation stopping behavior")
        elif row["stop_reason"] == "eos" and tokens[-1] not in report["eos_ids"]:
            raise ValueError("EOS token was not saved")
        elif row["stop_reason"] == "length_limit" and len(tokens) != MAX_NEW_TOKENS:
            raise ValueError("Output shorter than cap without EOS")
        if any(not isinstance(t, int) or not 0 <= t < report["vocab_size"] for t in tokens):
            raise ValueError("Invalid generated token ID")
        quality, empty = summary_quality(row["text"], case["references"], row["stop_reason"])
        if row["rouge_l_f1"] != quality or row["empty_summary"] != empty:
            raise ValueError("Summary quality does not recompute")
        times = [row[key] for key in ("switch_ms", "ttft_ms", "request_ms")]
        if row["ttft_ms"] is None:
            if tokens or row["stop_reason"] != "nonfinite_logits":
                raise ValueError("Missing first-token timing")
            times[1] = times[0]
        if any(not math.isfinite(t) for t in times) or not 0 <= times[0] <= times[1] <= times[2]:
            raise ValueError("Timing boundaries differ")
        if not 0 <= row["logical_tensor_bytes"] <= LIVE_CAP:
            raise ValueError("Logical tensor cap exceeded")
        if row["working_kv_bytes"] <= 0 or row["logical_tensor_bytes"] != (
            report["fixed_gpu_tensor_bytes"] + row["working_kv_bytes"]
        ) or not row["peak_reserved_bytes"] >= row["peak_allocated_bytes"] >= row["logical_tensor_bytes"]:
            raise ValueError("GPU memory accounting differs")
        if job["method"] != "native" and (cycle == 0 and position == 0 or cycle in AUDIT_AGES):
            if not isinstance(row["weight_digest"], str) or len(row["weight_digest"]) != 64:
                raise ValueError("Missing scheduled complete mutable-state digest")
            if job["method"] in EXACT and row["weight_digest"] != report["canonical_target_digest"]:
                raise ValueError("An exact method failed full mutable-state verification")
    return records


def summarize(root, scope):
    workload = read_json(root / "inputs/workload.json")
    jobs = schedule(scope)
    stages = {}
    for job in jobs:
        folder = root / "stages" / job["key"]
        report = read_json(folder / "report.json")
        records = audit_stage(job, workload, rows(folder / "requests.jsonl"), report)
        stages[job["key"]] = (records, report)
    references = {
        method: {row["id"]: row for row in stages[f"{scope}_resident_{method}"][0]}
        for method in ("host_copy", "native")
    }
    comparisons = []
    divergent = []
    for job in jobs:
        if job["resident"]:
            continue
        records, report = stages[job["key"]]
        method = job["method"]
        ref_method = "native" if method == "native" else "host_copy"
        comparison = []
        for row in records:
            exact = references["host_copy"][row["id"]]
            own_ref = references[ref_method][row["id"]]
            canonical_diff = first_difference(row["generated_ids"], exact["generated_ids"])
            history_diff = first_difference(row["generated_ids"], own_ref["generated_ids"])
            if row["stop_reason"] == "nonfinite_logits":
                canonical_diff = canonical_diff if canonical_diff is not None else len(row["generated_ids"]) + 1
                history_diff = history_diff if history_diff is not None else len(row["generated_ids"]) + 1
            comparison.append((row, exact, canonical_diff, history_diff))
            if canonical_diff is not None or history_diff is not None:
                divergent.append(dict(job=job["key"], id=row["id"], cycle=row["cycle"],
                    first_difference_from_canonical=canonical_diff, first_difference_from_own_path=history_diff,
                    completed_switches=row["completed_switches"], from_task=row["from_task"],
                    method_tokens=len(row["generated_ids"]), canonical_tokens=len(exact["generated_ids"]),
                    rouge_l_f1=row["rouge_l_f1"], canonical_rouge_l_f1=exact["rouge_l_f1"],
                    empty_summary=row["empty_summary"], stop_reason=row["stop_reason"]))
        eligible = [row for row in records if row["ttft_ms"] is not None]
        same_work = [row for row, _, diff, _ in comparison if diff is None]
        comparisons.append(dict(
            job=job, boot_id=report["boot_id"], dialogues=len(records), own_reference=ref_method,
            sequence_agreement_vs_canonical=sum(diff is None for _, _, diff, _ in comparison),
            sequence_agreement_vs_own_reference=sum(diff is None for _, _, _, diff in comparison),
            mean_rouge_l_f1=mean(row["rouge_l_f1"] for row in records),
            canonical_mean_rouge_l_f1=mean(ref["rouge_l_f1"] for _, ref, _, _ in comparison),
            empty_summaries=sum(row["empty_summary"] for row in records),
            capped=sum(row["stop_reason"] == "length_limit" for row in records),
            nonfinite=sum(row["stop_reason"] == "nonfinite_logits" for row in records),
            audited_weight_states=sum(row["weight_digest"] is not None for row in records),
            audited_canonical_matches=sum(row["weight_digest"] is not None and
                row["weight_digest"] == report["canonical_target_digest"] for row in records),
            generated_tokens=sum(len(row["generated_ids"]) for row in records),
            preceding_outbound_switch_ms=metrics([row["preceding_outbound_switch_ms"] for row in records]),
            switch_ms=metrics([row["switch_ms"] for row in records]),
            ttft_ms=metrics([row["ttft_ms"] for row in eligible]) if eligible else None,
            request_ms=metrics([row["request_ms"] for row in records]),
            identical_to_canonical_request_ms=metrics([row["request_ms"] for row in same_work]) if same_work else None,
            correction_bytes=report["correction_bytes"], fixed_gpu_tensor_bytes=report["fixed_gpu_tensor_bytes"],
            peak_allocated_bytes=report["peak_allocated_bytes"], peak_reserved_bytes=report["peak_reserved_bytes"],
        ))
    block_pairs = []
    for block in sorted({item["job"]["block"] for item in comparisons}):
        by_method = {item["job"]["method"]: item for item in comparisons if item["job"]["block"] == block}
        direct, shared = by_method["direct"], by_method["shared"]
        direct_rows, _ = stages[direct["job"]["key"]]
        shared_rows, _ = stages[shared["job"]["key"]]
        if any(a["id"] != b["id"] or a["cycle"] != b["cycle"] for a, b in zip(direct_rows, shared_rows)):
            raise ValueError("Direct/Shared paired requests differ")
        same_boot = direct["boot_id"] == shared["boot_id"]
        matched = [(a, b) for a, b in zip(direct_rows, shared_rows) if a["generated_ids"] == b["generated_ids"]]
        comparable = matched if same_boot else []
        block_pairs.append(dict(block=block,
            direct_and_shared_same_boot=same_boot,
            full_block_same_boot=len({x["boot_id"] for x in by_method.values()}) == 1,
            identical_output_sequences=len(matched), paired_timing_requests=len(comparable),
            correction_bytes_saved=direct["correction_bytes"]-shared["correction_bytes"],
            shared_minus_direct_ttft_ms=mean(b["ttft_ms"]-a["ttft_ms"]
                                            for a, b in comparable) if comparable else None,
            shared_minus_direct_request_ms=mean(b["request_ms"]-a["request_ms"]
                                               for a, b in comparable) if comparable else None))
    exact_outputs = all(item["sequence_agreement_vs_canonical"] == item["dialogues"]
                        for item in comparisons if item["job"]["method"] in EXACT)
    native_matches = all(item["sequence_agreement_vs_own_reference"] == item["dialogues"]
                         for item in comparisons if item["job"]["method"] == "native")
    canonical_rows = references["host_copy"].values()
    canonical_empty = sum(row["empty_summary"] for row in canonical_rows)
    if not exact_outputs:
        status = "REVIEW_CANONICAL_GENERATION"
    elif not native_matches:
        status = "REVIEW_NATIVE_REFERENCE"
    elif scope == "pilot" and canonical_empty:
        status = "REVIEW_GENERATION_FORMAT"
    else:
        status = "COMPLETED"
    summary = dict(
        status=status,
        task=TASK, scope=scope, distinct_dialogues=len(canonical_rows),
        timed_return_requests=sum(item["dialogues"] for item in comparisons),
        canonical_resident_mean_rouge_l_f1=mean(row["rouge_l_f1"] for row in canonical_rows),
        canonical_resident_empty_summaries=canonical_empty,
        canonical_resident_capped=sum(row["stop_reason"] == "length_limit" for row in canonical_rows),
        native_resident_mean_rouge_l_f1=mean(row["rouge_l_f1"] for row in references["native"].values()),
        native_output_sequences_equal_to_own_reference=native_matches,
        canonical_output_sequences_equal=exact_outputs, paired_direct_shared=block_pairs,
        resident_reference_boots={method: stages[f"{scope}_resident_{method}"][1]["boot_id"]
                                  for method in references},
        comparisons=comparisons, divergent_outputs=divergent,
        quality_metric=QUALITY_METRIC,
        quality_scope="Macro mean over dialogues; within each dialogue, mean ROUGE-L F1 over its official references",
        timing_scope="Single-user serialized return, including return switch and generation; outgoing interventions separate",
    )
    write_json(root / f"{scope}_summary.json", summary)
    return summary
