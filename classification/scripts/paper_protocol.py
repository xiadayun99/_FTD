"""Request schedules shared by the public audit and GPU replay."""
from copy import deepcopy
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from confirmation_protocol import METHODS, jobs

TASKS = ("mrpc", "qnli", "agnews")
FREQUENCY_METHODS = ("shared", "direct", "native")
QUANTA = {"low": 128, "medium": 8, "high": 1}
SWITCHES = {"low": 2, "medium": 47, "high": 383}
CAP = 32 * 2**30
KV_CAP = 512 * 2**20


def confirmation_jobs():
    return [job for job in jobs() if "block" in job]


def frequency_jobs():
    regimes = tuple(QUANTA)
    return [
        dict(key=f"r{block}_{method}", phase="trace", tag=f"r{block}_{method}",
             method=method, block=block, regime=regime)
        for block, order in enumerate(itertools.permutations(FREQUENCY_METHODS))
        for regime in regimes[block % 3:] + regimes[:block % 3]
        for method in order
    ]


def switch_count(tasks):
    current = TASKS[0]
    count = 0
    for task in tasks:
        count += int(task != current)
        current = task
    return count


def frequency_order(cases, regime):
    quantum = QUANTA[regime]
    by_task = {task: [case for case in cases if case["task"] == task] for task in TASKS}
    if len(cases) != 384 or any(len(task_cases) != 128 for task_cases in by_task.values()):
        raise ValueError("Expected 128 questions for each of three tasks")
    ordered = [
        case
        for start in range(0, 128, quantum)
        for task in TASKS
        for case in by_task[task][start:start + quantum]
    ]
    if switch_count(case["task"] for case in ordered) != SWITCHES[regime]:
        raise ValueError("Frequency schedule switch count differs")
    return ordered


def reorder_workload(workload, regime):
    result = deepcopy(workload)
    result["check"] = frequency_order(result["check"], regime)
    for position, case in enumerate(result["check"]):
        case["original_position"] = case["position"]
        case["position"] = position
    return result

