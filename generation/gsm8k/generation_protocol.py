"""Frozen GSM8K cohort, return schedule, and CPU audit for the ACL experiment."""

from __future__ import annotations

from decimal import Decimal
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

TASK = "gsm8k"
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
PROTOCOL_VERSION = "ftd_acl2027_gsm8k_return_v1"
DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
# Published by openai/gsm8k in dataset_infos.json at 25fb0412dbcc2e5128322234d829fbd6ece72c7c.
DATA = {
    "train": (7473, 4166206, "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "test": (1319, 749738, "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
}
INSTRUCTION = (
    "Solve the following math word problem. Show concise reasoning. "
    "On the final line, write #### followed by the numerical answer.\n\n"
)
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
ANSWER = re.compile(r"^[ \t]*####[ \t]*\$?(" + NUMBER + r")[ \t]*\.?[ \t]*$", re.MULTILINE)
CALCULATOR = re.compile(r"<<[^<>]*>>")


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


def numeric_answer(text):
    """Score only an explicit final marker. Preserve exact decimal equality."""
    matches = ANSWER.findall(text)
    if not matches:
        return None
    value = Decimal(matches[-1].replace(",", ""))
    if value == 0:
        return "0"
    fixed = format(value, "f")
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


def question_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def indexed(split, examples):
    result = []
    for index, example in enumerate(examples):
        question = example["question"].strip()
        answer = example["answer"].strip()
        gold = numeric_answer(answer)
        if not question:
            raise ValueError(f"Invalid GSM8K row {split}:{index}")
        result.append(dict(id=f"{split}:{index}", question=question, gold=gold,
                           question_sha256=question_key(question),
                           completion=CALCULATOR.sub("", answer)))
    return result


def unique_order(examples, seed, excluded):
    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    seen = set(excluded)
    result = []
    for row in ordered:
        key = row["question_sha256"]
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def select_cohorts(train, test, train_count=TRAIN_COUNT, test_count=TEST_COUNT,
                   dev_count=DEV_COUNT, warmup_count=WARMUP_COUNT):
    train = indexed("train", train)
    test = indexed("test", test)
    train_keys = {row["question_sha256"] for row in train}
    test_keys = {row["question_sha256"] for row in test}
    train_order = unique_order([row for row in train if row["gold"] is not None], SEED, test_keys)
    test_order = unique_order([row for row in test if row["gold"] is not None], SEED + 1, train_keys)
    end_dev = train_count + dev_count
    end_warmup = end_dev + warmup_count
    if len(train_order) < end_warmup or len(test_order) < test_count:
        raise ValueError("Too few independent GSM8K rows after question-level deduplication")
    return dict(
        train=train_order[:train_count], dev=train_order[train_count:end_dev],
        warmup=train_order[end_dev:end_warmup], test=test_order[:test_count],
    )


def official_file(root, split, data_dir=None):
    nrows, size, expected = DATA[split]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{split}.jsonl"

    def check(candidate):
        if candidate.stat().st_size != size or sha(candidate) != expected:
            raise ValueError(f"Expected the pinned official GSM8K {split}.jsonl: {candidate}")

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
                    request = Request(url, headers={"User-Agent": "FTD-GSM8K-research"})
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
                    f"Could not download {url}. Put the official train.jsonl and test.jsonl "
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


def prompt_ids(tokenizer, question):
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    if not isinstance(result, list) or not result or any(type(token) is not int for token in result):
        raise TypeError("Expected one Qwen3 chat-template token sequence")
    return result


def encode_training(tokenizer, item):
    prompt = prompt_ids(tokenizer, item["question"])
    target = tokenizer.encode(item["completion"], add_special_tokens=False)
    if not target or tokenizer.eos_token_id is None:
        raise ValueError("Expected a nonempty completion and a Qwen3 EOS token")
    target = target + [tokenizer.eos_token_id]
    tokens = prompt + target
    if len(tokens) > MAX_CONTEXT:
        raise ValueError(f"Training sequence exceeds frozen context: {item['id']}")
    return tokens, [-100] * len(prompt) + target


def prepare_data(root, model_dir, data_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False, local_files_only=True)
    train = official_file(root / "dataset_source", "train", data_dir)
    test = official_file(root / "dataset_source", "test", data_dir)
    cohorts = select_cohorts(train, test)
    training = [encode_training(tokenizer, row) for row in cohorts["train"]]
    training_plan = [dict(id=row["id"], question_sha256=row["question_sha256"],
                          tokens=tokens, labels=labels)
                     for row, (tokens, labels) in zip(cohorts["train"], training)]
    workload = {}
    for key in ("dev", "warmup", "test"):
        cases = []
        for item in cohorts[key]:
            tokens = prompt_ids(tokenizer, item["question"])
            if len(tokens) + MAX_NEW_TOKENS > MAX_CONTEXT:
                raise ValueError(f"Evaluation sequence exceeds frozen context: {item['id']}")
            cases.append(dict(id=item["id"], gold=item["gold"], prompt_ids=tokens,
                              question_sha256=item["question_sha256"], prompt_sha256=identity(tokens)))
        workload[key] = cases
    write_json(root / "inputs/training.json", training_plan)
    write_json(root / "inputs/workload.json", workload)
    write_json(root / "inputs/cohort.json", dict(
        protocol=PROTOCOL_VERSION, source_url=DATA_URL,
        source_sha256={key: meta[2] for key, meta in DATA.items()},
        train_seed=SEED, test_seed=SEED + 1, distinct_question_counts={
            key: len(value) for key, value in cohorts.items()},
        normalized_question_overlap=0, official_test_seen_by_sft=False,
        unparseable_source_answers={"train": sum(numeric_answer(x["answer"]) is None for x in train),
                                    "test": sum(numeric_answer(x["answer"]) is None for x in test)},
        max_training_tokens=max(len(row["tokens"]) for row in training_plan),
        max_evaluation_prompt_tokens=max(len(row["prompt_ids"]) for key in workload for row in workload[key]),
        training_sha256=sha(root / "inputs/training.json"),
        workload_sha256=sha(root / "inputs/workload.json"),
    ))


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
                      gold=case["gold"], cycle=cycle, completed_switches=2 * cycle,
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
        prediction = None if row["stop_reason"] == "nonfinite_logits" else numeric_answer(row["text"])
        if row["prediction"] != prediction or row["correct"] != (prediction == case["gold"]):
            raise ValueError("Task accuracy does not recompute")
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
            if job["method"] in EXACT and row["weight_digest"] != report["canonical_gsm8k_digest"]:
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
                    method_tokens=len(row["generated_ids"]), canonical_tokens=len(exact["generated_ids"]),
                    prediction=row["prediction"], canonical_prediction=exact["prediction"], gold=row["gold"],
                    correct=row["correct"], canonical_correct=exact["correct"]))
        eligible = [row for row in records if row["ttft_ms"] is not None]
        same_work = [row for row, _, diff, _ in comparison if diff is None]
        comparisons.append(dict(
            job=job, boot_id=report["boot_id"], questions=len(records), own_reference=ref_method,
            sequence_agreement_vs_canonical=sum(diff is None for _, _, diff, _ in comparison),
            sequence_agreement_vs_own_reference=sum(diff is None for _, _, _, diff in comparison),
            correct=sum(row["correct"] for row in records),
            reference_correct=sum(ref["correct"] for _, ref, _, _ in comparison),
            unparsed=sum(row["prediction"] is None for row in records),
            capped=sum(row["stop_reason"] == "length_limit" for row in records),
            nonfinite=sum(row["stop_reason"] == "nonfinite_logits" for row in records),
            audited_weight_states=sum(row["weight_digest"] is not None for row in records),
            audited_canonical_matches=sum(row["weight_digest"] is not None and
                row["weight_digest"] == report["canonical_gsm8k_digest"] for row in records),
            generated_tokens=sum(len(row["generated_ids"]) for row in records),
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
        block_pairs.append(dict(block=block,
            direct_and_shared_same_boot=direct["boot_id"] == shared["boot_id"],
            full_block_same_boot=len({x["boot_id"] for x in by_method.values()}) == 1,
            identical_output_sequences=sum(a["generated_ids"] == b["generated_ids"]
                                           for a, b in zip(direct_rows, shared_rows)),
            correction_bytes_saved=direct["correction_bytes"]-shared["correction_bytes"],
            shared_minus_direct_ttft_ms=mean(b["ttft_ms"]-a["ttft_ms"]
                                            for a, b in zip(direct_rows, shared_rows)),
            shared_minus_direct_request_ms=mean(b["request_ms"]-a["request_ms"]
                                               for a, b in zip(direct_rows, shared_rows))))
    exact_outputs = all(item["sequence_agreement_vs_canonical"] == item["questions"]
                        for item in comparisons if item["job"]["method"] in EXACT)
    canonical_unparsed = sum(row["prediction"] is None for row in references["host_copy"].values())
    status = "COMPLETED" if exact_outputs else "REVIEW_CANONICAL_GENERATION"
    if exact_outputs and scope == "pilot" and canonical_unparsed > len(references["host_copy"]) // 2:
        status = "REVIEW_GENERATION_FORMAT"
    summary = dict(
        status=status,
        scope=scope, distinct_questions=len(references["host_copy"]),
        timed_return_requests=sum(item["questions"] for item in comparisons),
        canonical_resident_correct=sum(row["correct"] for row in references["host_copy"].values()),
        canonical_resident_unparsed=canonical_unparsed,
        canonical_resident_capped=sum(row["stop_reason"] == "length_limit"
                                      for row in references["host_copy"].values()),
        native_resident_correct=sum(row["correct"] for row in references["native"].values()),
        canonical_output_sequences_equal=exact_outputs, paired_direct_shared=block_pairs,
        resident_reference_boots={method: stages[f"{scope}_resident_{method}"][1]["boot_id"]
                                  for method in references},
        comparisons=comparisons, divergent_outputs=divergent,
        accuracy_scope="GSM8K final-answer exact match; repeated rounds share the same questions",
        timing_scope="Single-user serialized return, including return switch and generation; outgoing interventions separate",
    )
    write_json(root / f"{scope}_summary.json", summary)
    return summary
