"""Bound preparation memory without changing the frozen correction arithmetic.

References are explicit file-backed CPU views. The original encoder runs once
per projection; its result is spooled before the next projection. The final
payload uses the standard safetensors layout and the original Patch objects.
No file I/O or heap trimming is added to the measured request path.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np

MIB = 2**20
SIZES = {"I32":4, "F32":4, "I16":2, "F16":2, "BF16":2, "U8":1}
DTYPES = {"I32":"int32", "F32":"float32", "I16":"int16", "F16":"float16", "BF16":"bfloat16", "U8":"uint8"}
FIELD_TYPES = {"packed":"U8", "indices":"I32", "values":"I16"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(MIB), b""):
            h.update(part)
    return h.hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def memory_snapshot():
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":",1)
        if key in ("MemAvailable", "MemTotal", "SwapFree", "SwapTotal"):
            out[key+"_bytes"] = int(value.split()[0])*1024
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
                key,value = line.split(":",1);out[key+"_bytes"] = int(value.split()[0])*1024
    except FileNotFoundError:
        pass
    return out


def trim_heap():
    """Setup only: release unreachable Python buffers and free glibc arenas."""
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t];trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


class PreparationResourceLimit(MemoryError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = details


class TensorFile:
    """Validated header plus one mapped tensor at a time; never caches tensors."""
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            first = f.read(8)
            if len(first) != 8:
                raise ValueError("Truncated tensor file")
            length = struct.unpack("<Q", first)[0]
            if not 2 <= length <= min(100_000_000, self.path.stat().st_size-8):
                raise ValueError("Invalid safetensors header length")
            def unique(pairs):
                result = {}
                for key,value in pairs:
                    if key in result:
                        raise ValueError("Duplicate safetensors key")
                    result[key] = value
                return result
            self.header = json.loads(f.read(length), object_pairs_hook=unique)
        self.start = 8+length
        self.records = {k:v for k,v in self.header.items() if k != "__metadata__"}
        spans = []
        for name, row in self.records.items():
            code = row["dtype"];shape = row["shape"];a,b = row["data_offsets"]
            if (code not in SIZES or any(type(n) is not int or n < 0 for n in shape)
                    or type(a) is not int or type(b) is not int
                    or not 0 <= a <= b <= self.path.stat().st_size-self.start
                    or b-a != math.prod(shape)*SIZES[code]):
                raise ValueError("Invalid tensor span: "+name)
            if b>a:
                spans.append((a,b))
        cursor = 0
        for a,b in sorted(spans):
            if a != cursor:
                raise ValueError("Tensor byte buffer overlaps or contains a hole")
            cursor = b
        if cursor != self.path.stat().st_size-self.start:
            raise ValueError("Unindexed tensor data")

    def raw(self, name):
        a,b = self.records[name]["data_offsets"]
        if a == b:
            return np.empty(0,dtype=np.uint8)
        return np.memmap(self.path,dtype=np.uint8,mode="c",offset=self.start+a,shape=(b-a,))

    def tensor(self, torch, name, cast=None):
        row = self.records[name]
        dtype = getattr(torch, DTYPES[row["dtype"]])
        if math.prod(row["shape"]) == 0:
            value = torch.empty(tuple(row["shape"]),dtype=dtype)
        else:
            value = torch.from_numpy(self.raw(name)).view(dtype).reshape(tuple(row["shape"]))
        return value if cast is None or value.dtype == cast else value.to(dtype=cast)


class MappedTensors(Mapping):
    def __init__(self, torch, file, names=None, cast=None):
        self.torch = torch;self.file = TensorFile(file);self.cast = cast
        self.names = tuple(sorted(self.file.records if names is None else names))
        if not set(self.names) <= set(self.file.records):
            raise ValueError("Mapped reference coverage differs")

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, name):
        if name not in self.names:
            raise KeyError(name)
        return self.file.tensor(self.torch,name,self.cast)


def tensor_digest(torch, values):
    """Same name/shape/dtype/raw-byte digest as frozen_kv_engine, streamed."""
    h = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
        cpu = value.detach().to("cpu").contiguous().view(torch.uint8)
        h.update(memoryview(cpu.numpy()))
        del cpu,value
    return h.hexdigest()


def load_targets(torch, e, directory):
    metadata = e.read_signed(Path(directory)/"target_metadata.json")
    targets = {}
    for task in ("mrpc", "qnli", "agnews"):
        mapped = MappedTensors(torch,Path(directory)/"reference_slices"/(task+".safetensors"))
        if any(row["dtype"] != "F16" for row in mapped.file.records.values()):
            raise ValueError("Canonical references must be serialized FP16")
        actual = tensor_digest(torch,mapped)
        if actual != metadata["target_slice_sha256"][task] or actual != e.LOCAL_TARGET_HASHES[task]:
            raise ValueError("Full canonical reference identity differs: "+task)
        targets[task] = mapped
        trim_heap()
    return targets,metadata


class TensorStream:
    """Data grouped by element width; write header and streams with bounded RAM."""
    def __init__(self, directory):
        self.directory = Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.files = {};self.lengths = {};self.records = {};self.total = 0

    def add(self, name, code, shape, raw):
        if name in self.records or code not in SIZES:
            raise ValueError("Duplicate tensor or unsupported dtype")
        shape = list(shape)
        if any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("Invalid tensor shape")
        view = memoryview(raw)
        if view.nbytes != math.prod(shape)*SIZES[code]:
            raise ValueError("Tensor byte count differs")
        data = view.cast("B") if view.nbytes else memoryview(b"")
        if code not in self.files:
            self.files[code] = (self.directory/(code+".bin")).open("wb")
            self.lengths[code] = 0
        begin = self.lengths[code]
        self.files[code].write(data)
        self.lengths[code] += len(data);self.total += len(data)
        self.records[name] = dict(dtype=code,shape=shape,local_offsets=[begin,self.lengths[code]])

    def finish(self, destination):
        for f in self.files.values():
            f.close()
        order = sorted(self.lengths,key=lambda k:(-SIZES[k],k))
        bases = {};offset = 0
        for code in order:
            bases[code] = offset;offset += self.lengths[code]
        header = {}
        for name,row in self.records.items():
            start,end = row["local_offsets"];base = bases[row["dtype"]]
            header[name] = dict(dtype=row["dtype"],shape=row["shape"],data_offsets=[base+start,base+end])
        encoded = json.dumps(header,separators=(",",":"),ensure_ascii=False).encode()
        encoded += b" "*((-len(encoded))%8)
        destination = Path(destination)
        partial = destination.with_name(destination.name+".partial")
        with partial.open("wb") as out:
            out.write(struct.pack("<Q",len(encoded)));out.write(encoded)
            for code in order:
                with (self.directory/(code+".bin")).open("rb") as f:
                    shutil.copyfileobj(f,out,length=MIB)
        partial.replace(destination)
        TensorFile(destination)  # Independent span/hole/header check.
        self.close()

    def close(self):
        for f in self.files.values():
            if not f.closed:
                f.close()
        shutil.rmtree(self.directory,ignore_errors=True)


class SpoolPatch:
    def __init__(self, torch, metadata, file):
        self.torch = torch;self.info = metadata;self.file = TensorFile(file)

    @property
    def nbytes(self):
        return self.info["bytes"]

    def metadata(self):
        return dict(self.info)

    def __getattr__(self, name):
        if name in FIELD_TYPES:
            return self.file.tensor(self.torch,name)
        raise AttributeError(name)


class SpoolingEncoder:
    def __init__(self, torch, original, directory, progress, fixed_bytes, cap_bytes, floor_bytes=512*MIB):
        self.torch = torch;self.original = original;self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.progress = Path(progress);self.fixed = fixed_bytes;self.cap = cap_bytes;self.floor = floor_bytes
        self.count = 0;self.payload = 0

    def __call__(self, predictor, target):
        before = memory_snapshot()
        scratch_allowance = 64*predictor.numel()+64*MIB
        if before["MemAvailable_bytes"] < self.floor+scratch_allowance:
            raise PreparationResourceLimit("Insufficient setup headroom before encoding the next projection",
                dict(index=self.count,limit_kind="setup_headroom",estimated_scratch_allowance_bytes=scratch_allowance,**before))
        patch = self.original(predictor,target)  # Unchanged codec/arithmetic.
        metadata = patch.metadata();self.payload += patch.nbytes
        row = dict(index=self.count,mode=patch.mode,shape=metadata["shape"],patch_bytes=patch.nbytes,
                   cumulative_payload_bytes=self.payload,fixed_plus_payload_bytes=self.fixed+self.payload,
                   logical_cap_bytes=self.cap,monotonic_ns=time.perf_counter_ns())
        row.update(memory_snapshot())
        with self.progress.open("a") as out:
            out.write(json.dumps(row)+"\n");out.flush()
        if self.fixed+self.payload > self.cap:
            raise PreparationResourceLimit("Recorded model/factor/descriptor/correction bytes exceed the registered cap",dict(row,limit_kind="logical_tensor_budget"))
        file = self.directory/(f"patch_{self.count:05d}.safetensors")
        stream = TensorStream(self.directory/(f"stream_{self.count:05d}"))
        try:
            for field,code in FIELD_TYPES.items():
                value = getattr(patch,field).detach().cpu().contiguous()
                stream.add(field,code,list(value.shape),value.numpy())
                del value
            stream.finish(file)
        finally:
            stream.close()
        del patch
        trim_heap()
        memory = memory_snapshot()
        if memory["MemAvailable_bytes"] < self.floor:
            raise PreparationResourceLimit("Available memory fell below the setup floor; stop before exhaustion",dict(row,after_trim=memory))
        self.count += 1
        if self.count%64 == 0:
            print(f"Encoded/spooled {self.count} projections; payload={self.payload/MIB:.2f} MiB; available={memory['MemAvailable_bytes']/MIB:.1f} MiB",flush=True)
        return SpoolPatch(self.torch,metadata,file)

    @contextmanager
    def install(self, module):
        original = module.encode
        module.encode = self
        try:
            yield
        finally:
            module.encode = original


def serialize(directory,banks,contract):
    directory = Path(directory)
    stream = TensorStream(directory/"serialization_streams")
    meta = dict(contract=contract,banks={})
    try:
        for kind,tasks in banks.items():
            meta["banks"][kind] = {}
            for task,patches in tasks.items():
                meta["banks"][kind][task] = {}
                for i,(name,patch) in enumerate(sorted(patches.items())):
                    key = f"{kind}.{task}.{i}"
                    for field,code in FIELD_TYPES.items():
                        value = getattr(patch,field).detach().cpu().contiguous()
                        stream.add(key+"."+field,code,list(value.shape),value.numpy())
                        del value
                    meta["banks"][kind][task][name] = dict(key=key,metadata=patch.metadata())
        stream.finish(directory/"patches.safetensors")
    finally:
        stream.close()
    meta["payload_sha256"] = sha(directory/"patches.safetensors")
    dump(directory/"patch_metadata.json",meta)
    return meta


def deserialize(directory,torch,patch_class,contract,device="cuda:0"):
    directory = Path(directory)
    meta = json.loads((directory/"patch_metadata.json").read_text())
    file = directory/"patches.safetensors"
    if meta["contract"] != contract or sha(file) != meta["payload_sha256"]:
        raise ValueError("Correction contract/payload differs")
    source = TensorFile(file);banks = {}
    for kind,tasks in meta["banks"].items():
        banks[kind] = {}
        for task,patches in tasks.items():
            banks[kind][task] = {}
            for name,info in patches.items():
                key = info["key"];m = info["metadata"]
                fields = [source.tensor(torch,key+"."+f) for f in FIELD_TYPES]
                stats = {k:m[k] for k in ("elements","changed","escapes","changed_fraction","dense_reference_bytes")}
                patch = patch_class(tuple(m["shape"]),m["dtype"],m["mode"],*fields,
                                    m["predictor_sha256"],m["target_sha256"],stats).to(device)
                banks[kind][task][name] = patch
                del fields,patch
    trim_heap()
    return banks,meta


def check_roundtrip(torch,core,rp,directory,contract):
    """On-device test against the frozen safetensors loader and CUDA decoder."""
    directory = Path(directory);directory.mkdir()
    a = torch.ones(4099,dtype=torch.float16)
    b = a.clone();b[0] = -0.;b[-1] = 1e-4
    patch = rp.encode(a,b)
    raw = np.full(4099,15000,dtype=np.int16)
    target = raw.astype(np.int32)+np.resize(np.array([-1,0,1]),raw.size);target[-1] = 30000
    c = torch.from_numpy(raw.copy()).view(torch.float16)
    d = torch.from_numpy(target.astype(np.int16)).view(torch.float16)
    banks = dict(merge=dict(fixture=dict(sparse=patch,two_bit=rp.encode(c,d),
                                      zero=rp.encode(a,a),dense=rp.encode(a,-a))))
    serialize(directory,banks,contract)
    frozen,_ = core.deserialize_patches(directory,"cuda:0",contract)
    mapped,_ = deserialize(directory,torch,rp.Patch,contract)
    for name,start,expected in [("sparse",a,b),("two_bit",c,d),("zero",a,a),("dense",a,-a)]:
        for loaded in (frozen,mapped):
            probe = start.cuda();loaded["merge"]["fixture"][name].apply_(probe,verify_input=True)
            torch.cuda.synchronize()
            if not rp.bit_equal(probe,expected):
                raise ValueError("Streamed payload roundtrip differs: "+name)
    result = dict(all_bits_equal=True,codec_cases=4,loaders_checked=2,
                  frozen_loader_interoperability=True,large_model_checked=False)
    dump(directory/"roundtrip_check.json",result)
    return result
