"""Standard safetensors checkpoints and streamed file hashes, outside timing."""
from pathlib import Path
import shutil

import disk_protocol as sp


def require_space(directory,required):
    free=shutil.disk_usage(directory).free
    if free<required:
        raise OSError('Need %.2f GiB free disk here; found %.2f GiB: %s' %
                      (required/sp.GIB,free/sp.GIB,directory))
    return free


def save_tensors(tensors,path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    if path.exists() or partial.exists():raise FileExistsError('Refusing to overwrite a tensor checkpoint: '+str(path))
    payload=sum(v.numel()*v.element_size() for v in tensors.values())
    require_space(path.parent,payload+sp.DISK_FLOOR+16*sp.MIB)
    save_file(tensors,str(partial))
    with safe_open(str(partial),framework='pt',device='cpu') as f:
        if set(f.keys())!=set(tensors):raise ValueError('Saved tensor coverage differs')
        for name,value in tensors.items():
            if tuple(f.get_slice(name).get_shape())!=tuple(value.shape):raise ValueError('Saved tensor shape differs')
    partial.replace(path)
    return dict(file_sha256=sp.sha(path),file_bytes=path.stat().st_size,tensor_bytes=payload)


def cpu_slice(torch,e,directory,name):
    """Materialize a verified CPU copy before any timed host-copy control."""
    from safetensors.torch import load_file
    record=sp.read(directory/(name+'.json'));file=directory/(name+'.safetensors')
    if sp.sha(file)!=record['file_sha256']:raise ValueError('Reference file changed: '+name)
    raw=load_file(str(file),device='cpu')
    values={key:value.detach().clone().contiguous() for key,value in raw.items()}
    del raw
    if any(value.dtype!=torch.float16 for value in values.values()):raise ValueError('Reference dtype differs')
    if e.tensor_digest(torch,values.items())!=record['tensor_sha256']:raise ValueError('Reference tensor hash differs: '+name)
    return values,record


def reference_bundle(torch,e,root):
    directory=root/'artifacts/references';metadata=sp.read(directory/'metadata.json')
    targets={}
    for task in sp.TASKS:
        targets[task],record=cpu_slice(torch,e,directory,task)
        if record['shared_state_sha256']!=metadata['shared_state_sha256']:
            raise ValueError('Reference common-state identity differs')
        if record['tensor_sha256']!=metadata['target_slice_sha256'][task]:
            raise ValueError('Reference target identity differs')
    base,record=cpu_slice(torch,e,directory,'base')
    if record['tensor_sha256']!=metadata['base_slice_sha256']:raise ValueError('Base slice differs')
    if any(set(values)!=set(base) for values in targets.values()):raise ValueError('Reference projection coverage differs')
    e.LOCAL_TARGET_HASHES=dict(metadata['target_slice_sha256'])
    return targets,base,metadata


def save_banks(directory,banks,contract):
    directory.mkdir(parents=True,exist_ok=True)
    tensors={};meta=dict(contract=contract,banks={})
    for kind,owners in banks.items():
        meta['banks'][kind]={}
        for owner,values in owners.items():
            meta['banks'][kind][owner]={}
            for i,(name,patch) in enumerate(sorted(values.items())):
                key=f'{kind}.{owner}.{i}'
                for field in ('packed','indices','values'):
                    tensors[key+'.'+field]=getattr(patch,field).detach().cpu().contiguous()
                meta['banks'][kind][owner][name]=dict(key=key,metadata=patch.metadata())
    saved=save_tensors(tensors,directory/'patches.safetensors')
    meta.update(payload_sha256=saved['file_sha256'],file_bytes=saved['file_bytes'],tensor_bytes=saved['tensor_bytes'])
    sp.dump(directory/'patch_metadata.json',meta)
    return meta


def load_banks(torch,patch_class,directory,contract):
    from safetensors.torch import load_file
    meta=sp.read(directory/'patch_metadata.json');file=directory/'patches.safetensors'
    if meta['contract']!=contract or sp.sha(file)!=meta['payload_sha256']:
        raise ValueError('Correction arithmetic contract or payload changed')
    raw=load_file(str(file),device='cpu');banks={};seen=set();total=0
    for kind,owners in meta['banks'].items():
        banks[kind]={}
        for owner,values in owners.items():
            banks[kind][owner]={}
            for name,entry in values.items():
                key=entry['key'];m=entry['metadata'];fields=[]
                for field in ('packed','indices','values'):
                    label=key+'.'+field;seen.add(label);fields.append(raw[label])
                stats={k:m[k] for k in ('elements','changed','escapes','changed_fraction','dense_reference_bytes')}
                patch=patch_class(tuple(m['shape']),m['dtype'],m['mode'],*fields,
                    m['predictor_sha256'],m['target_sha256'],stats).to('cuda:0')
                if patch.nbytes!=m['bytes']:raise ValueError('Correction byte accounting differs')
                banks[kind][owner][name]=patch;total+=patch.nbytes
    if seen!=set(raw) or total!=meta['tensor_bytes']:raise ValueError('Correction checkpoint coverage differs')
    return banks,meta


def checkpoint_outputs(root,paths):
    """Record relative files; reject paths outside the run and any symlink."""
    outputs={}
    for path in paths:
        path=Path(path)
        if not path.is_file() or path.is_symlink():raise ValueError('Missing or linked checkpoint: '+str(path))
        rel=path.relative_to(root)
        if path.resolve().is_relative_to(root.resolve()) is False:raise ValueError('Checkpoint escapes result directory')
        outputs[str(rel)]=sp.sha(path)
    if not outputs:raise ValueError('A completed phase must have outputs')
    return outputs


def outputs_match(root,outputs):
    if not isinstance(outputs,dict) or not outputs:return False
    for rel,expected in outputs.items():
        name=Path(rel)
        if name.is_absolute() or '..' in name.parts:return False
        path=root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):return False
        if sp.sha(path)!=expected:return False
    return True
