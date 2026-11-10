"""Calibrate target-shared repairs from canonical sources and LoRA factors.

No old per-edge correction bank is read or built. The four measured serving
modules remain byte-for-byte unchanged. CPU references are reused separately
from the newly measured correction preparation cost.
"""
from pathlib import Path
import hashlib
import os
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor/runtime'))
import disk_protocol as sp
import disk_store as ds


def raw_digest(shape, dtype, raw):
    raw = np.asarray(raw)
    if raw.dtype != np.uint16 or not raw.flags.c_contiguous:
        raise ValueError('Contiguous uint16 raw codes required')
    if raw.size != int(np.prod(shape)):
        raise ValueError('Raw shape differs')
    h = hashlib.sha256()
    h.update(str(tuple(shape)).encode()); h.update(str(dtype).encode())
    h.update(raw.tobytes())
    return h.hexdigest()


def exception_indices(target, predictor):
    target, predictor = np.asarray(target), np.asarray(predictor)
    if target.dtype != np.uint16 or predictor.dtype != np.uint16 or target.shape != predictor.shape:
        raise ValueError('Target/predictor raw layouts differ')
    if target.size > np.iinfo(np.int32).max:
        raise ValueError('Projection exceeds the admitted int32 index geometry')
    delta = target.reshape(-1).astype(np.int32) - predictor.reshape(-1).astype(np.int32)
    return np.flatnonzero((delta < -1) | (delta > 1)).astype(np.int32)


def build(root, source):
    import shared_engine as se
    import disk_engine as engine
    import run_8b_disk as legacy_driver
    import bounded_io as bio
    from shared_codec import encode_target
    began = time.perf_counter()
    # with_banks=False is essential: only model, factors and CPU references.
    session = engine.Session(source, legacy_driver.port_module(), 'direct', with_banks=False)
    if session.banks is not None: raise ValueError('Unexpected old correction bank')
    torch = session.torch
    import rounding_patch as rp
    torch.cuda.synchronize()
    setup = time.perf_counter() - began
    directory = root / 'artifacts/shared_target'; directory.mkdir(parents=True, exist_ok=False)
    names = sorted(session.names); params = dict(session.model.named_parameters())
    if len(names) != 252: raise ValueError('Expected 252 mutable projections')
    # Match SharedPatchGroup's actual descriptor tensor layout; no directed banks.
    descriptor_bytes = 3 * (4*8*len(names) + 3*4*len(names) + 2*(len(names)+1)*4)
    fixed = session.e.model_payload_bytes(session.model) + session.factors.bytes + descriptor_bytes
    banks = {'target': {}}; records = []; payload = 0; predictors = 0
    calibration_began = time.perf_counter()
    with torch.no_grad():
        for target in sp.TASKS:
            banks['target'][target] = {}
            incoming = {src: session.factors.projection_values(src, target) for src in sp.TASKS if src != target}
            if any(set(values) != set(names) for values, sign in incoming.values()):
                raise ValueError('Incoming projection coverage differs')
            for name in names:
                tensor = session.targets[target][name]
                raw = rp.bits(tensor)
                digest = raw_digest(tensor.shape, tensor.dtype, raw)
                groups = []; hashes = {}; incoming_counts = {}
                for src, (values, sign) in incoming.items():
                    a,b,scale = values[name]; w = params[name]
                    w.copy_(session.targets[src][name])
                    w.addmm_(b,a,beta=1.,alpha=sign*scale)
                    if not bool(torch.isfinite(w).all()):
                        raise FloatingPointError('Nonfinite standalone predictor')
                    prediction = rp.bits(w)
                    hashes[src] = raw_digest(w.shape,w.dtype,prediction)
                    indices = exception_indices(raw,prediction)
                    groups.append(indices); incoming_counts[src] = int(indices.size)
                    predictors += 1
                    del prediction
                encoded = encode_target(raw,groups)
                patch = se.make_patch(torch,tensor,encoded,hashes,digest)
                banks['target'][target][name] = patch; payload += patch.nbytes
                if fixed + payload > sp.CAP: raise MemoryError('Shared fixed state exceeds 32 GiB')
                records.append(dict(target=target,name=name,**patch.metadata()))
                sp.append(directory/'progress.jsonl',dict(projections=len(records),predictors=predictors,
                    payload_bytes=payload,incoming_escape_counts=incoming_counts,memory=bio.memory_snapshot()))
                del raw,groups,encoded
                if len(records)%36 == 0:
                    session.port.require_no_observed_foreign(os.getpgrp());bio.trim_heap()
                    print('CALIBRATE_SHARED:',len(records),'/756',flush=True)
    torch.cuda.synchronize(); calibration_seconds = time.perf_counter()-calibration_began
    if predictors != 1512 or len(records) != 756: raise ValueError('Standalone coverage differs')
    serialize_began = time.perf_counter()
    metadata = ds.save_banks(directory,banks,se.extended_contract(session.contract))
    if metadata['tensor_bytes'] != payload: raise ValueError('Saved payload accounting differs')
    serialization_seconds = time.perf_counter()-serialize_began
    sp.dump(directory/'records.json',records)
    sp.dump(directory/'build_report.json',dict(status='BUILT_PENDING_FRESH_GPU_ADMISSION',
        standalone_correction_build=True,old_edge_bank_read=False,old_edge_bank_generated=False,
        new_gemm_calibration_performed=True,projections=756,calibrated_predictors=1512,
        payload_bytes=payload,new_payload_sha256=metadata['payload_sha256'],
        fixed_state_before_corrections_bytes=fixed,descriptor_bytes=descriptor_bytes,
        session_setup_wall_seconds=setup,factor_preparation_seconds=session.factor_preparation_seconds,
        calibration_wall_seconds=calibration_seconds,serialization_wall_seconds=serialization_seconds,
        correction_build_wall_seconds=calibration_seconds+serialization_seconds,
        worker_total_wall_seconds=time.perf_counter()-began,
        reference_construction_and_adapter_training_included=False,
        mode_counts={mode:sum(x['mode']==mode for x in records) for mode in ('target_low2','dense')},
        runtime_contract=se.extended_contract(session.contract)))
    return [p for p in directory.rglob('*') if p.is_file()]


def worker(root, source, job):
    if job['phase'] == 'build': return build(root, source)
    import shared_engine as se
    if job['phase'] == 'vectors': return se.vectors(root, source)
    if job['phase'] == 'trace': return se.trace(root, source, 'shared', 'admit_shared', True)
    raise ValueError('Unknown standalone phase')
