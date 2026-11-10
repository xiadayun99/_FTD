"""Build one new adapter and run the generation protocol on the frozen operators."""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import random
import sys
import time

import generation_protocol as gp

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "vendor/runtime")]
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def boot_id():
    return gp.read("/proc/sys/kernel/random/boot_id").strip()


def configure_paths(config):
    import disk_protocol as sp
    sp.MODEL_DIR = Path(config["model_dir"])
    sp.TASKS = gp.TASKS
    return sp


def configure_port():
    from run_8b_disk import port_module

    port = port_module()
    port.TASKS = gp.TASKS
    original_contract = port.runtime_contract

    def arithmetic_contract(torch, engine, profile, numerical_profile):
        import pynvml as nv
        value = original_contract(torch, engine, profile, numerical_profile)
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU to CUDA")
        nv.nvmlInit()
        try:
            if nv.nvmlDeviceGetCount() != 1:
                raise ValueError("This run requires one GPU visible to both NVML and CUDA")
            uuid = nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(0))
        finally:
            nv.nvmlShutdown()
        value["gpu_uuid"] = uuid.decode("ascii") if isinstance(uuid, bytes) else uuid
        # The boot ID is recorded with the stage. Arithmetic identity and full
        # edge/cycle admission, not the boot ID, determine safe bank reuse.
        del value["boot_id"]
        return value

    def full_path_checks(torch, engine, params, targets, make):
        checks = []
        for source, target in itertools.permutations(gp.TASKS, 2):
            engine.install_host_target(torch, params, targets[source])
            state = make(source)
            state.apply(target)
            torch.cuda.synchronize()
            actual = engine.tensor_digest(torch, [(n, params[n]) for n in targets[target]])
            if actual != engine.LOCAL_TARGET_HASHES[target]:
                raise ValueError(f"Full grouped edge differs: {source} -> {target}")
            checks.append(dict(source=source, target=target, sha256=actual))
            del state
        engine.install_host_target(torch, params, targets[gp.TASK])
        state = make(gp.TASK)
        for _ in range(3):
            for target in (gp.PEERS[0], gp.TASK, gp.PEERS[1], gp.TASK):
                state.apply(target)
                torch.cuda.synchronize()
                actual = engine.tensor_digest(torch, [(n, params[n]) for n in targets[target]])
                if actual != engine.LOCAL_TARGET_HASHES[target]:
                    raise ValueError(f"Continuous grouped path differs at {target}")
                checks.append(dict(target=target, cycle=True, sha256=actual))
        return checks

    port.runtime_contract = arithmetic_contract
    port.full_path_checks = full_path_checks
    return port


def verify_gpu(root, port):
    import disk_engine as engine
    import numpy as np
    from disk_training import collate
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from shared_codec import encode_target
    from shared_engine import make_patch
    from shared_group import SharedPatchGroup
    from standalone_build import exception_indices

    torch, core, frozen_engine, _, _, _, contract = engine.runtime(port, root, initial=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The single GSM8K adapter is trained with a BF16 backbone")
    vectors = core.validate_cuda_decoder_vectors(torch)
    raw = np.tile(np.array([1, 100, 201, 65534, 500, 512, 500, 512], dtype=np.uint16), 1024)
    first, second = raw.copy(), raw.copy()
    first[0::8] = 0
    first[2] = 400
    second[1::8] = 101
    second[5] = 900
    union = [exception_indices(raw, predictor) for predictor in (first, second)]
    encoded = encode_target(raw, union)
    if encoded["mode"] != "target_low2":
        raise AssertionError("Shared vector must exercise the packed mode")
    target = torch.from_numpy(raw.view(np.float16).copy()).to("cuda:0")
    work = torch.empty_like(target)
    patch = make_patch(torch, target, encoded, {}, "synthetic GPU decoder check").to("cuda:0")
    group = SharedPatchGroup({"weight": work}, {"weight": patch})
    for predictor in (first, second):
        work.copy_(torch.from_numpy(predictor.view(np.float16).copy()))
        group.apply_()
        torch.cuda.synchronize()
        if not torch.equal(work.view(torch.int16), target.view(torch.int16)):
            raise ValueError("Shared grouped CUDA decoder failed an incoming predictor")
    # Exercise BF16 LoRA loss/backprop and FP16 request-local KV generation
    # before allocating the real model or starting the 96-update SFT.
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        pad_token_id=0, eos_token_id=2)
    config._attn_implementation = "eager"
    tiny = Qwen3ForCausalLM(config).to(device="cuda:0", dtype=torch.bfloat16)
    toy_lora = LoraConfig(r=2, lora_alpha=4, target_modules=list(PROJECTIONS), task_type="CAUSAL_LM")
    model = get_peft_model(tiny, toy_lora)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [param for param in model.parameters() if param.requires_grad]
    for param in trainable:
        param.data = param.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=gp.LEARNING_RATE)
    loss = model(**collate(torch, [([4, 5, 6, 2], [-100, -100, 6, 2])], 0)).loss
    if not math.isfinite(float(loss.detach().item())):
        raise FloatingPointError("Tiny BF16 LoRA gradient check did not produce finite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.add_adapter("second", toy_lora)
    model.set_adapter("second")
    model.set_adapter("default")
    model.eval().to(dtype=torch.float16)
    tail = frozen_engine.tail_argument(model)
    if tail is None:
        raise ValueError("Tiny PEFT model lacks the pinned tail-logit API")
    tokens, _, _, _, cache = gen_tokens(torch, model, "cuda:0", tail, [4, 5, 6], {2}, limit=3)
    if not tokens or cache.get_seq_length() < 3:
        raise ValueError("Tiny autoregressive generation failed")
    gp.write_json(root / "runtime_contract.json", contract)
    gp.write_json(root / "gpu_probe.json", dict(status="VERIFIED", boot_id=boot_id(),
        direct_decoder_vectors=vectors, shared_grouped_decoder_predictors=2,
        tiny_bf16_lora_update=True, tiny_fp16_native_generated_tokens=len(tokens), contract=contract))


def train(root, port, model_dir):
    import disk_engine as engine
    from disk_training import collate, load_base, save_fp16_adapter
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    torch, _, _, _, _, _, contract = engine.runtime(port, root)
    examples = gp.read_json(root / "inputs/training.json")
    if len(examples) != gp.TRAIN_COUNT:
        raise ValueError("GSM8K training cohort changed")
    torch.manual_seed(gp.SEED)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = load_base(torch, model_dir, "cuda:0", torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=gp.RANK, lora_alpha=gp.ALPHA, lora_dropout=.05, bias="none",
        task_type="CAUSAL_LM", target_modules=list(PROJECTIONS),
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA factors")
    for param in trainable:
        param.data = param.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=gp.LEARNING_RATE, weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    log = root / "training.jsonl"
    total_updates = gp.TRAIN_COUNT * gp.EPOCHS // gp.ACCUMULATION
    updates = seen = 0
    losses = []
    began = time.perf_counter()
    for epoch in range(gp.EPOCHS):
        order = list(range(len(examples)))
        random.Random(gp.SEED + epoch).shuffle(order)
        for index in order:
            example = examples[index]
            if seen % 64 == 0:
                port.require_no_observed_foreign(os.getpgrp())
            batch = collate(torch, [(example["tokens"], example["labels"])], pad)
            loss = model(**batch).loss
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite supervised LoRA training loss")
            (loss / gp.ACCUMULATION).backward()
            del loss, batch
            losses.append(value)
            seen += 1
            if seen % gp.ACCUMULATION == 0:
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                gp.append_row(log, dict(epoch=epoch, update=updates, examples_seen=seen,
                    mean_loss=sum(losses)/len(losses), gradient_norm=float(norm.item())))
                losses.clear()
                if updates % 4 == 0 or updates == total_updates:
                    print(f"TRAIN_GSM8K: {updates}/{total_updates} updates", flush=True)
    if losses or updates != total_updates:
        raise ValueError("Training update accounting differs")
    model.eval()
    model.config.use_cache = True
    folder = root / "trained/lora" / gp.TASK
    folder.mkdir(parents=True, exist_ok=False)
    save_fp16_adapter(torch, model, folder)
    gp.write_json(folder / "training_contract.json", dict(
        status="TRAINED", boot_id=boot_id(), contract=contract, rank=gp.RANK,
        alpha=gp.ALPHA, microbatch=1, accumulation=gp.ACCUMULATION, epochs=gp.EPOCHS,
        optimizer="AdamW", learning_rate=gp.LEARNING_RATE, updates=updates, examples_seen=seen,
        dataset_sha256=gp.sha(root / "inputs/training.json"), final_checkpoint=True,
        train_seconds=time.perf_counter()-began, backbone_dtype="bfloat16",
        training_adapter_dtype="float32", saved_adapter_dtype="float16"))


def gen_tokens(torch, model, device, tail_arg, prompt_ids, eos_ids, limit, allow_nonfinite=False):
    """Single-request DynamicCache and greedy decoding with one host sync per token."""
    tokens = []
    first_ns = None
    cache = None
    reason = "length_limit"
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for step in range(limit):
            output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, **{tail_arg: 1})
            cache = output.past_key_values
            if cache is None:
                raise ValueError("Missing request-local DynamicCache")
            logits = output.logits[0, -1]
            token = int(torch.where(torch.isfinite(logits).all(), logits.argmax(), -1).item())
            now = time.perf_counter_ns()
            del logits, output
            if token < 0:
                if not allow_nonfinite:
                    raise FloatingPointError("Nonfinite generated-token logits")
                reason = "nonfinite_logits"
                break
            if step == 0:
                first_ns = now
            tokens.append(token)
            if token in eos_ids:
                reason = "eos"
                break
            if step + 1 < limit:
                input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    finished_ns = time.perf_counter_ns()
    steps = len(tokens) + int(reason == "nonfinite_logits")
    if cache is None or int(cache.get_seq_length()) != len(prompt_ids) + steps - 1:
        raise ValueError("Request-local KV length does not match the greedy decode")
    return tokens, reason, first_ns, finished_ns, cache


class ReturnState:
    """An independent one-slot worker using the released transition operators."""

    def __init__(self, root, port, method):
        import disk_engine as engine
        from compact_group import CompactPatchGroup
        from direct_transition import DirectSwitch
        from shared_engine import extended_contract, load_shared, validate_shared
        from shared_group import SharedSwitch

        base_method = "direct" if method in ("direct", "shared", "incremental") else (
            "reference" if method == "host_copy" else "native")
        session = engine.Session(root, port, base_method, with_banks=method == "direct")
        if len(session.names) != 252:
            raise ValueError("Expected seven mutable projections per Qwen3-8B layer")
        torch = session.torch
        self.session, self.torch, self.model, self.method = session, torch, session.model, method
        self.params = dict(self.model.named_parameters())
        self.transition = None
        self.validation = None
        if method == "shared":
            session.contract = extended_contract(session.contract)
            session.banks, session.bank_metadata = load_shared(torch, root, session.contract)
            self.validation = validate_shared(session)
        elif method == "direct":
            self.validation = session.validate(True)
        self.restore_initial()
        if method == "shared":
            self.transition = SharedSwitch(torch, self.params, session.factors, session.banks,
                                           gp.TASK, session.contract)
        elif method == "direct":
            self.transition = DirectSwitch(torch, self.params, session.factors, session.banks,
                                           gp.TASK, session.contract, CompactPatchGroup)
        self.correction_bytes = sum(patch.nbytes for owners in (session.banks or {}).values()
                                    for patches in owners.values() for patch in patches.values())
        self.fixed_bytes = (session.e.model_payload_bytes(self.model)
                            + (session.factors.bytes if session.factors is not None else 0)
                            + self.correction_bytes
                            + (self.transition.auxiliary_bytes if self.transition is not None else 0))
        self.tail_arg = session.e.tail_argument(self.model)
        if self.tail_arg is None:
            raise ValueError("Last-token logits API unavailable")
        cfg = self.model.config
        heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
        self.bytes_per_kv_token = 2 * 2 * cfg.num_hidden_layers * heads * head_dim
        if self.fixed_bytes + gp.MAX_CONTEXT * self.bytes_per_kv_token > gp.LIVE_CAP:
            raise MemoryError(f"{method} exceeds the 32 GiB model/factors/corrections/working-KV budget")

    def restore_initial(self):
        if self.method == "native":
            self.model.set_adapter(gp.TASK)
        else:
            self.session.e.install_host_target(self.torch, self.params, self.session.targets[gp.TASK])
        if self.transition is not None and (self.transition.task != gp.TASK or not self.transition.valid):
            raise ValueError("Expected intact canonical GSM8K state at reset")
        self.active = gp.TASK
        self.torch.cuda.synchronize()
        if self.method != "native" and self.digest() != self.session.e.LOCAL_TARGET_HASHES[gp.TASK]:
            raise ValueError("Initial mutable state does not match the GSM8K reference")

    def digest(self):
        if self.method == "native":
            return None
        return self.session.e.tensor_digest(self.torch, [(n, self.params[n]) for n in self.session.names])

    def switch(self, task):
        if task == self.active:
            return
        if self.transition is not None:
            self.transition.apply(task)
        elif self.method == "incremental":
            self.session.factors.add(self.params, self.active, task)
        elif self.method == "host_copy":
            self.session.e.install_host_target(self.torch, self.params, self.session.targets[task])
        elif self.method == "native":
            self.model.set_adapter(task)
        else:
            raise ValueError(f"Unknown transition method {self.method}")
        self.active = task


def trace(root, job, port, model_dir):
    from transformers import AutoTokenizer

    folder = root / "stages" / job["key"]
    folder.mkdir(parents=True, exist_ok=False)
    workload = gp.read_json(root / "inputs/workload.json")
    cases = gp.return_cases(workload, job["scope"], job["block"])
    if job["resident"]:
        cases = [(item, 0) for item, _ in cases]
    state = ReturnState(root, port, job["method"])
    torch = state.torch
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    eos = state.model.generation_config.eos_token_id
    eos_ids = frozenset(([eos] if isinstance(eos, int) else eos or []) + [tokenizer.eos_token_id])
    if not eos_ids or None in eos_ids:
        raise ValueError("Qwen3 generation EOS contract differs")
    for index, task in enumerate(gp.PEERS):
        if not job["resident"]:
            state.switch(task)
            state.switch(gp.TASK)
        _, _, _, _, warm_cache = gen_tokens(torch, state.model, device, state.tail_arg,
            workload["warmup"][index]["prompt_ids"], eos_ids, limit=8)
        del warm_cache
    state.restore_initial()
    torch.cuda.reset_peak_memory_stats(device)
    canonical_digest = state.session.e.LOCAL_TARGET_HASHES[gp.TASK]
    if state.validation is not None:
        gp.write_json(folder / "weight_admission.json", state.validation)
    current_cycle = 0
    peak_allocated = peak_reserved = 0
    for position, (case, cycle) in enumerate(cases):
        outbound_ms = 0.
        hidden_cycle_ms = 0.
        if cycle:
            for skipped in range(current_cycle + 1, cycle):
                began = time.perf_counter_ns()
                state.switch(gp.peer(skipped))
                state.switch(gp.TASK)
                torch.cuda.synchronize()
                hidden_cycle_ms += (time.perf_counter_ns() - began) / 1e6
            current_cycle = cycle
            from_task = gp.peer(cycle)
            began = time.perf_counter_ns()
            state.switch(from_task)
            torch.cuda.synchronize()
            outbound_ms = (time.perf_counter_ns() - began) / 1e6
        else:
            from_task = gp.TASK
        torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter_ns()
        state.switch(gp.TASK)
        if cycle:
            torch.cuda.synchronize()
        installed = time.perf_counter_ns()
        tokens, reason, first, finished, cache = gen_tokens(
            torch, state.model, device, state.tail_arg, case["prompt_ids"], eos_ids,
            gp.MAX_NEW_TOKENS, allow_nonfinite=job["method"] == "incremental",
        )
        working_bytes = state.session.e.cache_bytes(cache)
        request_allocated = int(torch.cuda.max_memory_allocated(device))
        request_reserved = int(torch.cuda.max_memory_reserved(device))
        peak_allocated = max(peak_allocated, request_allocated)
        peak_reserved = max(peak_reserved, request_reserved)
        del cache
        digest = state.digest() if (cycle in gp.AUDIT_AGES or cycle == 0 and position == 0) else None
        if job["method"] in gp.EXACT and digest is not None and digest != canonical_digest:
            raise ValueError(f"Exact mutable-weight digest differs after {cycle} return cycles")
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction = None if reason == "nonfinite_logits" else gp.numeric_answer(text)
        gp.append_row(folder / "requests.jsonl", dict(
            position=position, id=case["id"], prompt_sha256=case["prompt_sha256"], gold=case["gold"],
            cycle=cycle, from_task=from_task, completed_switches=2*cycle,
            generated_ids=tokens, text=text, prediction=prediction, correct=prediction == case["gold"],
            stop_reason=reason, weight_digest=digest, preceding_outbound_switch_ms=outbound_ms,
            extra_pilot_interventions_ms=hidden_cycle_ms, switch_ms=(installed-began)/1e6,
            ttft_ms=(first-began)/1e6 if first is not None else None,
            request_ms=(finished-began)/1e6, working_kv_bytes=working_bytes,
            logical_tensor_bytes=state.fixed_bytes+working_bytes,
            peak_allocated_bytes=request_allocated, peak_reserved_bytes=request_reserved,
        ))
        if position % 8 == 0:
            port.require_no_observed_foreign(os.getpgrp())
        if (position + 1) % 8 == 0 or position + 1 == len(cases):
            print(f"{job['key']}: {position + 1}/{len(cases)}", flush=True)
    report = dict(job=job, boot_id=boot_id(), requests=len(cases), eos_ids=sorted(eos_ids),
        vocab_size=int(state.model.config.vocab_size), canonical_gsm8k_digest=canonical_digest,
        correction_bytes=state.correction_bytes, fixed_gpu_tensor_bytes=state.fixed_bytes,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        gpu_isolation_verified=False, effective_arithmetic_contract=state.session.contract,
        no_cross_request_kv=True, num_return_cycles=current_cycle, warmup_reset_to_canonical=True,
        timed_batch_size=1, pretokenized_prompts=True, decode="greedy", max_new_tokens=gp.MAX_NEW_TOKENS)
    gp.write_json(folder / "report.json", report)


def run(root, job, attempt, config):
    from disk_engine import prepare_reference, assemble_references, Session
    from standalone_build import build
    import run_8b_disk as legacy_driver

    phase = job["phase"]
    model_dir = Path(config["model_dir"])
    if phase == "data":
        gp.prepare_data(root, model_dir, config.get("data_dir"))
        return
    configure_paths(config)
    port = configure_port()
    port.require_no_observed_foreign(os.getpgrp())
    with port.Telemetry(attempt):
        if phase == "gpu_probe":
            verify_gpu(root, port)
        elif phase == "train":
            train(root, port, model_dir)
        elif phase == "merge":
            prepare_reference(root, port, gp.TASK)
            assemble_references(root)
        elif phase == "build":
            if job["method"] == "shared":
                # Reuse the measured standalone builder with this run's
                # task names and per-GPU arithmetic contract.
                original_port = legacy_driver.port_module
                legacy_driver.port_module = lambda: port
                try:
                    build(root, root)
                finally:
                    legacy_driver.port_module = original_port
            else:
                Session(root, port, "direct", with_banks=False).build()
        elif phase == "trace":
            trace(root, job, port, model_dir)
        else:
            raise ValueError(f"Unknown phase {phase}")
        port.require_no_observed_foreign(os.getpgrp())
    samples = gp.rows(attempt / "telemetry.jsonl")
    if not samples or any(row["observation"]["foreign_pids"] for row in samples):
        raise RuntimeError("GPU_ISOLATION_FAILED: observed foreign GPU work or no samples")
    if any(row["observation"]["backend"] != "nvml" or
           not row["observation"]["process_enumeration_verified"] for row in samples):
        raise RuntimeError("GPU_ISOLATION_UNVERIFIED: NVML sampling unavailable")
    if phase == "trace":
        folder = root / "stages" / job["key"]
        report = gp.read_json(folder / "report.json")
        report["gpu_isolation_verified"] = True
        report["telemetry_samples"] = len(samples)
        gp.audit_stage(job, gp.read_json(root / "inputs/workload.json"),
                       gp.rows(folder / "requests.jsonl"), report)
        gp.write_json(folder / "report.json", report)
