import contextlib
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generation_protocol as gp
import generation_worker as worker
import run_generation as runner


def example(question, answer):
    return dict(question=question, answer=answer)


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True,
                               enable_thinking=False, return_dict=False)
        assert messages[0]["role"] == "user"
        return [3, 4, len(messages[0]["content"])]

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        assert "<<" not in text
        return [17, 18, 19]


class DataTests(unittest.TestCase):
    def test_numeric_answer(self):
        self.assertEqual(gp.numeric_answer("work 8\n#### $1,024.00\n"), "1024")
        self.assertEqual(gp.numeric_answer("#### -2\n#### +6.25."), "6.25")
        self.assertEqual(gp.numeric_answer("#### -0.000"), "0")
        self.assertIsNone(gp.numeric_answer("We got 12"))
        self.assertIsNone(gp.numeric_answer("#### 2 apples"))
        self.assertEqual(gp.numeric_answer("#### 1.00000000000000000000000000000001"),
                         "1.00000000000000000000000000000001")

    def test_deterministic_no_leakage(self):
        train = [example(f"Train {i}", f"Work <<1+1=2>>\n#### {i}") for i in range(13)]
        train += [example(" Shared  prompt ", "#### 1"), example("bad", "no marker")]
        test = [example(f"Test {i}", f"#### {i}") for i in range(6)]
        test.append(example("shared prompt", "#### 3"))
        result = gp.select_cohorts(train, test, train_count=5, dev_count=4, warmup_count=2, test_count=5)
        self.assertEqual(result, gp.select_cohorts(
            train, test, train_count=5, dev_count=4, warmup_count=2, test_count=5))
        values = [item["question_sha256"] for group in result.values() for item in group]
        self.assertEqual(len(values), len(set(values)))
        self.assertNotIn(gp.question_key("shared prompt"), values)
        self.assertNotIn(gp.question_key("bad"), values)
        self.assertTrue(all(row["id"].startswith("test:") for row in result["test"]))

    def test_insufficient_independent_rows(self):
        sample = [example("identical", "#### 1")]
        with self.assertRaisesRegex(ValueError, "independent"):
            gp.select_cohorts(sample, sample, 1, 1, 1, 1)

    def test_labels_only_supervise_completion(self):
        item = gp.indexed("train", [example("1 + 1?", "Add <<1+1=2>>.\n#### 2")])[0]
        ids, labels = gp.encode_training(Tokenizer(), item)
        self.assertEqual(labels[:3], [-100, -100, -100])
        self.assertEqual(labels[3:], ids[3:])
        self.assertEqual(ids[-1], 2)

    def test_local_source_copy_checks_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = (json.dumps(example("Test", "#### 0")) + "\n").encode()
            (base / "train.jsonl").write_bytes(data)
            meta = (1, len(data), hashlib.sha256(data).hexdigest())
            with patch.dict(gp.DATA, {"train": meta}):
                downloaded = gp.official_file(base / "downloaded", "train", base)
                self.assertEqual(downloaded[0]["question"], "Test")
                (base / "downloaded/train.jsonl").write_bytes(data.replace(b"Test", b"Fail"))
                with self.assertRaisesRegex(ValueError, "pinned official"):
                    gp.official_file(base / "downloaded", "train", base)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        self.index = index
        return self

    def all(self):
        return all(self.value)

    def argmax(self):
        return max(range(len(self.value)), key=self.value.__getitem__)

    def item(self):
        return self.value


class FakeTorch:
    long = int
    inference_mode = staticmethod(contextlib.nullcontext)

    @staticmethod
    def tensor(data, dtype, device):
        return data

    @staticmethod
    def isfinite(vector):
        return FakeTensor([math.isfinite(x) for x in vector.value])

    @staticmethod
    def where(cond, yes, no):
        return FakeTensor(yes if cond else no)


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeModel:
    def __init__(self, logits):
        self.logits = iter(logits)
        self.called = []

    def __call__(self, *, input_ids, past_key_values, use_cache, logits_to_keep):
        assert use_cache and logits_to_keep == 1
        length = len(input_ids[0]) + (past_key_values.get_seq_length() if past_key_values else 0)
        self.called.append((input_ids[0], length))
        return SimpleNamespace(logits=FakeTensor(next(self.logits)), past_key_values=FakeCache(length))


class GenerationTests(unittest.TestCase):
    def test_eos_is_retained_and_kv_is_request_local(self):
        model = FakeModel([[0, 2, 1], [0, 1, 2]])
        ids, reason, first, end, cache = worker.gen_tokens(
            FakeTorch, model, "cpu", "logits_to_keep", [4, 5, 6], {2}, limit=8)
        self.assertEqual(ids, [1, 2])
        self.assertEqual(reason, "eos")
        self.assertLessEqual(first, end)
        self.assertEqual(model.called, [([4, 5, 6], 3), ([1], 4)])
        self.assertEqual(cache.get_seq_length(), 4)

    def test_equal_logits_use_first_id_and_cap(self):
        model = FakeModel([[1, 1, 0]])
        ids, reason, *_ = worker.gen_tokens(
            FakeTorch, model, "cpu", "logits_to_keep", [4, 5], {2}, limit=1)
        self.assertEqual(ids, [0])
        self.assertEqual(reason, "length_limit")

    def test_nonfinite_allowed_only_for_uncorrected_control(self):
        model = FakeModel([[0, 2, 1], [0, float("nan"), 2]])
        ids, reason, first, *_ = worker.gen_tokens(
            FakeTorch, model, "cpu", "logits_to_keep", [4, 5], {2}, limit=4, allow_nonfinite=True)
        self.assertEqual(ids, [1])
        self.assertEqual(reason, "nonfinite_logits")
        self.assertIsNotNone(first)
        with self.assertRaises(FloatingPointError):
            worker.gen_tokens(FakeTorch, FakeModel([[0, float("inf"), 1]]),
                              "cpu", "logits_to_keep", [4, 5], {2}, limit=1)

    def test_incremental_switch_uses_actual_active_source(self):
        calls = []
        state = worker.ReturnState.__new__(worker.ReturnState)
        state.method = "incremental"
        state.active = gp.TASK
        state.transition = None
        state.params = {}
        state.session = SimpleNamespace(factors=SimpleNamespace(
            add=lambda params, source, target: calls.append((source, target))))
        for target in ("mrpc", "gsm8k", "qnli", "gsm8k", "gsm8k"):
            state.switch(target)
        self.assertEqual(calls, [("gsm8k", "mrpc"), ("mrpc", "gsm8k"),
                                 ("gsm8k", "qnli"), ("qnli", "gsm8k")])
        self.assertEqual(state.active, gp.TASK)


def simulated_workload():
    def make(count, prefix):
        return [dict(id=f"{prefix}:{i}", gold="1", prompt_ids=[3, i+4],
                     prompt_sha256=gp.identity([3, i+4])) for i in range(count)]
    return dict(dev=make(8, "dev"), warmup=make(2, "warmup"), test=make(128, "test"))


def simulated_stage(job, workload, corrupt_canonical=False, missing_answers=False):
    expected = gp.return_cases(workload, job["scope"], job["block"])
    records = []
    for position, (case, age) in enumerate(expected):
        age = 0 if job["resident"] else age
        variant = ((job["method"] == "native") or
                   (job["method"] == "incremental" and age >= 8 and position == 2) or
                   (corrupt_canonical and job["method"] == "shared" and position == 0))
        digest = None
        if job["method"] != "native" and (age == 0 and position == 0 or age in gp.AUDIT_AGES):
            digest = "b"*64 if job["method"] == "incremental" else "a"*64
        records.append(dict(
            position=position, id=case["id"], prompt_sha256=case["prompt_sha256"], cycle=age,
            gold=case["gold"], from_task=gp.peer(age) if age else gp.TASK, completed_switches=2*age,
            generated_ids=[7 if variant else 6, 2], text="#### 1" if not missing_answers else "bad",
            prediction="1" if not missing_answers else None, correct=not missing_answers,
            stop_reason="eos", weight_digest=digest, switch_ms=1., ttft_ms=2., request_ms=3.,
            working_kv_bytes=8, logical_tensor_bytes=108, peak_allocated_bytes=112, peak_reserved_bytes=116,
        ))
    report = dict(job=job, requests=len(records), boot_id="test-boot", gpu_isolation_verified=True,
        eos_ids=[2], vocab_size=256, canonical_gsm8k_digest="a"*64,
        fixed_gpu_tensor_bytes=100, peak_allocated_bytes=112, peak_reserved_bytes=116,
        correction_bytes={"shared": 7, "direct": 10}.get(job["method"], 0))
    return records, report


def write_simulated_run(root, scope, **kwargs):
    work = simulated_workload()
    gp.write_json(root / "inputs/workload.json", work)
    for job in gp.schedule(scope):
        records, report = simulated_stage(job, work, **kwargs)
        folder = root / "stages" / job["key"]
        gp.write_json(folder / "report.json", report)
        (folder / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


class AuditTests(unittest.TestCase):
    def test_confirmation_reverses_questions_and_order(self):
        work = simulated_workload()
        first = gp.return_cases(work, "confirm", 0)
        second = gp.return_cases(work, "confirm", 1)
        self.assertEqual([case for case, _ in first], [case for case, _ in second][::-1])
        self.assertEqual([cycle for _, cycle in second], list(range(1, 129)))
        methods = [job["method"] for job in gp.schedule("confirm") if not job["resident"]]
        self.assertEqual(methods[:5], methods[5:][::-1])
        self.assertEqual(len(gp.schedule("pilot")), 7)

    def test_stage_rejects_changed_timing_weight_and_identity(self):
        work = simulated_workload()
        job = next(job for job in gp.schedule("pilot") if job["method"] == "shared")
        for key, invalid in (("ttft_ms", 4.), ("weight_digest", "c"*64),
                             ("prompt_sha256", "bad"), ("peak_allocated_bytes", 107)):
            with self.subTest(key=key):
                records, report = simulated_stage(job, work)
                records[0][key] = invalid
                with self.assertRaises(ValueError):
                    gp.audit_stage(job, work, records, report)

    def test_nonfinite_result_kept_in_incremental_arm(self):
        work = simulated_workload()
        job = next(job for job in gp.schedule("pilot") if job["method"] == "incremental")
        records, report = simulated_stage(job, work)
        records[0].update(generated_ids=[], text="", prediction=None, correct=False,
                          stop_reason="nonfinite_logits", ttft_ms=None)
        gp.audit_stage(job, work, records, report)

    def test_real_summary_counts_unique_inputs_and_native_own_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_simulated_run(root, "confirm")
            summary = gp.summarize(root, "confirm")
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["distinct_questions"], 128)
            self.assertEqual(summary["timed_return_requests"], 1280)
            self.assertTrue(all(item["sequence_agreement_vs_canonical"] == 0
                                and item["sequence_agreement_vs_own_reference"] == 128
                                for item in summary["comparisons"] if item["job"]["method"] == "native"))
            self.assertEqual(summary["paired_direct_shared"][0]["correction_bytes_saved"], 3)

    def test_divergence_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_simulated_run(root, "pilot", corrupt_canonical=True)
            self.assertEqual(gp.summarize(root, "pilot")["status"], "REVIEW_CANONICAL_GENERATION")
            self.assertEqual(gp.first_difference([1, 2], [1, 2, 3]), 3)

    def test_pilot_gate_requires_parsable_canonical_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_simulated_run(root, "pilot", missing_answers=True)
            self.assertEqual(gp.summarize(root, "pilot")["status"], "REVIEW_GENERATION_FORMAT")


class ResumeTests(unittest.TestCase):
    def test_reuse_completed_and_detect_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "inputs/example.json"
            gp.write_json(path, {"example": 1})
            job = runner.PREPARE[0]
            gp.write_json(root / "completed/data.json", dict(
                job=job, outputs=runner.inventory(root, runner.outputs(root, job))))
            with patch.object(runner, "run_worker") as mocked:
                runner.phase(root, job, None, None)
                mocked.assert_not_called()
            gp.write_json(path, {"example": 2})
            with self.assertRaisesRegex(ValueError, "Completed phase changed"):
                runner.phase(root, job, None, None)

    def test_preserve_uncommitted_logs_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "artifacts/patches/direct"
            gp.write_json(bank / "build_report.json", {})
            (bank / "patches.safetensors").write_bytes(b"uncommitted")
            external = root / "external.safetensors"
            external.write_bytes(b"input")
            runner.incomplete(root, runner.PREPARE[4])
            self.assertFalse(bank.exists())
            self.assertEqual(external.read_bytes(), b"input")
            self.assertEqual(len(list((root / "interrupted").rglob("build_report.json"))), 1)
            self.assertFalse(list((root / "interrupted").rglob("*.safetensors")))

    def test_archive_excludes_tensor_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gp.write_json(root / "status.json", dict(status="STOPPED"))
            stage = root / "stages/failed"
            gp.write_json(stage / "report.json", {})
            (stage / "file.safetensors").write_bytes(b"large tensor")
            path = runner.archive(root, "pilot")
            with tarfile.open(path) as result:
                names = result.getnames()
            self.assertFalse(any(name.endswith(".safetensors") for name in names))
            self.assertTrue(any(name.endswith("status.json") for name in names))
            self.assertTrue(any(name.endswith("report.json") for name in names))

    def test_worker_failure_and_log_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gp.write_json(root / "run_config.json", {"model_dir": str(root)})
            with (root / ".run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                code, attempt = runner.run_worker(root, dict(key="invalid", phase="bad"),
                                                   None, 15, lock.fileno())
            self.assertNotEqual(code, 0)
            self.assertTrue((attempt / "worker.log").is_file())
            self.assertEqual(gp.read_json(attempt / "exit.json")["code"], code)

    def test_low_disk_preflight_can_retry_after_freeing_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "run"
            args = SimpleNamespace(output=root, model_dir=parent / "model",
                                   reference_run=parent / "refs", command="pilot")
            with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
                    patch.object(runner, "resolve_inputs") as resolve:
                self.assertEqual(runner.execute(args), 1)
                resolve.assert_not_called()
            with patch.object(runner.shutil, "disk_usage",
                              return_value=SimpleNamespace(free=gp.START_DISK)), \
                    patch.object(runner, "resolve_inputs", side_effect=FileNotFoundError("example")) as resolve:
                self.assertEqual(runner.execute(args), 1)
                resolve.assert_called_once()
            self.assertEqual(gp.read_json(root / "status.json")["stage"], "asset_preflight")

    def test_shared_builder_receives_run_specific_contract_port(self):
        import run_8b_disk as legacy
        import standalone_build

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            attempt.mkdir()
            original_port = legacy.port_module

            @contextlib.contextmanager
            def telemetry(folder):
                gp.append_row(folder / "telemetry.jsonl", dict(observation=dict(
                    backend="nvml", foreign_pids=[], process_enumeration_verified=True)))
                yield

            port = SimpleNamespace(Telemetry=telemetry, require_no_observed_foreign=lambda group: None)

            def fake_build(output, source):
                self.assertEqual(output, source)
                self.assertIs(legacy.port_module(), port)

            with patch.object(worker, "configure_port", return_value=port), \
                    patch.object(worker, "configure_paths"), patch.object(standalone_build, "build", fake_build):
                worker.run(root, dict(phase="build", method="shared"), attempt,
                           dict(model_dir=str(root / "model")))
            self.assertIs(legacy.port_module, original_port)


if __name__ == "__main__":
    unittest.main()
