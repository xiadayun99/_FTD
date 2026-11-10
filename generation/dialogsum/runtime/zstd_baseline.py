"""Zstd measurement primitives from the paper's measured CPU runner."""
import ctypes as C
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time

MIB, GIB = 2**20, 2**30
SIZES = {'BOOL':1, 'U8':1, 'I8':1, 'I16':2, 'U16':2, 'I32':4,
         'U32':4, 'I64':8, 'U64':8, 'F16':2, 'BF16':2, 'F32':4, 'F64':8}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path, progress=False):
    h, done = hashlib.sha256(), 0
    size = Path(path).stat().st_size
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(8*MIB)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress and (done//GIB != (done-len(block))//GIB or done == size):
                print('VERIFY_INPUT:', Path(path).name, '%.2f/%.2f GiB' % (done/GIB, size/GIB), flush=True)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')+b'\n')
    temporary.replace(path)


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key: '+key)
        out[key] = value
    return out


class TensorFile:
    """Read tensor byte ranges without importing torch or loading large tensors."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError('Expected a regular frozen tensor file: '+str(path))
        self.size = self.path.stat().st_size
        with self.path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Incomplete safetensors prefix')
            n = struct.unpack('<Q', prefix)[0]
            if not 2 <= n <= 64*MIB or n+8 > self.size:
                raise ValueError('Invalid safetensors header length')
            self.header_bytes = stream.read(n)
        self.offset = 8+n
        header = json.loads(self.header_bytes, object_pairs_hook=unique_pairs)
        if not isinstance(header, dict):
            raise ValueError('Tensor header is not an object')
        self.tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        cursor = 0
        for name, info in sorted(self.tensors.items(), key=lambda kv:tuple(kv[1]['data_offsets'])):
            dtype, shape = info['dtype'], info['shape']
            start, end = info['data_offsets']
            if dtype not in SIZES or not all(type(x) is int and x >= 0 for x in shape):
                raise ValueError('Unsupported tensor dtype or shape: '+name)
            if type(start) is not int or type(end) is not int or start != cursor or end < start:
                raise ValueError('Invalid tensor range: '+name)
            if end-start != math.prod(shape)*SIZES[dtype]:
                raise ValueError('Tensor range/shape mismatch: '+name)
            cursor = end
        if self.offset+cursor != self.size:
            raise ValueError('Tensor payload has holes or trailing bytes')
        self.payload_bytes = cursor

    def read_block(self, stream, name, offset, size):
        begin, end = self.tensors[name]['data_offsets']
        if offset < 0 or size < 0 or begin+offset+size > end:
            raise ValueError('Tensor block is outside its range')
        stream.seek(self.offset+begin+offset)
        data = stream.read(size)
        if len(data) != size:
            raise ValueError('Frozen input was truncated')
        return data


class Zstd:
    def __init__(self, level=3):
        name = ctypes.util.find_library('zstd') or 'libzstd.so.1'
        try:
            self.lib = C.CDLL(name)
        except OSError as exc:
            raise RuntimeError('libzstd is unavailable; no automatic installation or codec substitution was performed') from exc
        self.level = level
        definitions = {
            'ZSTD_compressBound':([C.c_size_t], C.c_size_t),
            'ZSTD_compress':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int],C.c_size_t),
            'ZSTD_decompress':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_size_t),
            'ZSTD_isError':([C.c_size_t],C.c_uint),
            'ZSTD_getErrorName':([C.c_size_t],C.c_char_p),
            'ZSTD_versionString':([],C.c_char_p)}
        for symbol, (args, result) in definitions.items():
            function = getattr(self.lib, symbol)
            function.argtypes, function.restype = args, result
        self.version = self.lib.ZSTD_versionString().decode()

    def checked(self, result):
        if self.lib.ZSTD_isError(result):
            raise ValueError(self.lib.ZSTD_getErrorName(result).decode())
        return result

    def compress(self, raw):
        capacity = self.lib.ZSTD_compressBound(len(raw))
        output = C.create_string_buffer(capacity)
        size = self.checked(self.lib.ZSTD_compress(output, capacity, C.c_char_p(raw), len(raw), self.level))
        return output.raw[:size]

    def decompress(self, frame, size):
        output = C.create_string_buffer(max(1, size))
        actual = self.checked(self.lib.ZSTD_decompress(output, size, C.c_char_p(frame), len(frame)))
        if actual != size:
            raise ValueError('Decoded byte length differs')
        return output.raw[:actual]


def xor_bytes(a, b):
    import numpy as np
    if len(a) != len(b):
        raise ValueError('XOR inputs have different lengths')
    return np.bitwise_xor(np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8)).tobytes()


def load_checkpoints(path, identity):
    rows = {}
    if not path.exists():
        return rows
    valid_end = 0
    partial = None
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                partial = line
                break
            record = json.loads(line)
            data = record['data']
            if (record['sha256'] != digest_bytes(canonical(data)) or
                    data['protocol_sha256'] != identity or data['key'] in rows):
                raise ValueError('An existing baseline checkpoint changed')
            rows[data['key']] = data
            valid_end = stream.tell()
    if partial is not None:
        (path.parent/'recovered_incomplete_tail.txt').write_bytes(partial)
        with path.open('r+b') as stream:
            stream.truncate(valid_end)
        print('RECOVERED: unfinished final metadata line; committed units retained', flush=True)
    return rows


def append_checkpoint(path, data):
    record = dict(data=data, sha256=digest_bytes(canonical(data)))
    with path.open('ab') as stream:
        stream.write(canonical(record)+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def measure_unit(codec, source, names, reference, block_bytes, repeats):
    result = dict(raw_bytes=0, zstd_frame_bytes=0, frame_prefix_bytes=0, blocks=0,
                  transform_encode_ms=0., compress_ms=0., cpu_restore_median_sum_ms=0.,
                  io_timed=False, verification_timed=False, raw_bytes_identical=True)
    with source.path.open('rb') as target_stream:
        ref_stream = reference.path.open('rb') if reference is not None else None
        try:
            for name in names:
                info = source.tensors[name]
                size = info['data_offsets'][1]-info['data_offsets'][0]
                if reference is not None:
                    other = reference.tensors.get(name)
                    if other is None or other['dtype'] != info['dtype'] or other['shape'] != info['shape']:
                        raise ValueError('Reference tensor layout differs: '+name)
                for offset in range(0, size, block_bytes):
                    n = min(block_bytes, size-offset)
                    raw = source.read_block(target_stream, name, offset, n)
                    base = reference.read_block(ref_stream, name, offset, n) if reference is not None else None
                    start = time.perf_counter_ns()
                    encoded = xor_bytes(raw, base) if base is not None else raw
                    transform_ns = time.perf_counter_ns()-start
                    start = time.perf_counter_ns()
                    frame = codec.compress(encoded)
                    compress_ns = time.perf_counter_ns()-start
                    timings = []
                    for repeat in range(repeats):
                        start = time.perf_counter_ns()
                        decoded = codec.decompress(frame, n)
                        restored = xor_bytes(decoded, base) if base is not None else decoded
                        timings.append((time.perf_counter_ns()-start)/1e6)
                        if restored != raw:
                            raise ValueError('Lossless restoration failed: '+name)
                    result['raw_bytes'] += n
                    result['zstd_frame_bytes'] += len(frame)
                    result['frame_prefix_bytes'] += 8
                    result['blocks'] += 1
                    result['transform_encode_ms'] += transform_ns/1e6
                    result['compress_ms'] += compress_ns/1e6
                    result['cpu_restore_median_sum_ms'] += statistics.median(timings)
        finally:
            if ref_stream is not None:
                ref_stream.close()
    return result


def summarize(plans, completed, config):
    groups={}
    for plan in plans:
        group=groups.setdefault(plan['group'],dict(raw_tensor_bytes=0,zstd_frame_bytes=0,
            container_prefix_and_header_bytes=0,frame_prefix_bytes=0,required_immutable_reference_file_bytes=0,
            side_metadata_bytes=0,compress_ms=0.,transform_encode_ms=0.,cpu_restore_median_sum_ms=0.,units=0,blocks=0))
        header=dict(plan['header'],codec='zstd',level=config['level'],block_bytes=config['block_bytes'])
        group['container_prefix_and_header_bytes']+=8+len(canonical(header))
        if plan['metadata'] is not None:
            group['side_metadata_bytes']+=plan['metadata'].stat().st_size
        if plan['reference'] is not None:
            group['required_immutable_reference_file_bytes']=plan['reference'].size
        for key,_ in plan['units']:
            record=completed[plan['group']+'/'+plan['label']+'/'+key]
            group['units']+=1
            group['raw_tensor_bytes']+=record['measurement']['raw_bytes']
            for name in ('zstd_frame_bytes','frame_prefix_bytes','blocks','compress_ms','transform_encode_ms','cpu_restore_median_sum_ms'):
                group[name]+=record['measurement'][name]
    for group in groups.values():
        group['computed_container_bytes']=group['zstd_frame_bytes']+group['frame_prefix_bytes']+group['container_prefix_and_header_bytes']
        group['coded_bytes_including_side_metadata']=group['computed_container_bytes']+group['side_metadata_bytes']
        group['coded_bytes_plus_required_reference']=group['coded_bytes_including_side_metadata']+group['required_immutable_reference_file_bytes']
        group['raw_recovery_verified']=True
        group['frame_bytes_divided_by_raw_tensor_bytes']=group['zstd_frame_bytes']/group['raw_tensor_bytes']
    return dict(status='LOSSLESS_BASELINE_COMPLETED',groups=groups,
        scope='CPU offline coding of mutable FP16 reference slices and existing patch tensors',
        codec='zstd',configuration=config,compressed_frames_measured_and_recovered=True,
        container_sizes_computed_from_measured_frames_and_explicit_framing=True,
        compressed_weight_files_retained=False,is_zipnn_or_ulpack_reproduction=False,
        gpu_serving_latency_measured=False,gpu_resident_memory_reduction_established=False,
        original_training_and_inference_repeated=False,submission_ready=False,
        timing_note='CPU compression and median-of-repeat restoration; input I/O and byte-equality checks excluded; no CPU-isolation guarantee')
