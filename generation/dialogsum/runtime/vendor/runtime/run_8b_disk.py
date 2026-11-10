#!/usr/bin/env python3
"""FTD 8B: tiny verification -> new training -> disk references/banks -> 18 stages."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import datetime as dt
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid

import disk_protocol as sp
import disk_store as ds
HERE=Path(__file__).resolve().parent


def verify_package():
    inventory=sp.read(HERE/'PACKAGE_MANIFEST.json')
    if not inventory:raise ValueError('Empty package manifest')
    for rel,expected in inventory.items():
        path=HERE/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink() or not path.is_file() or sp.sha(path)!=expected:
            raise ValueError('Package file changed: '+rel)
    pins=sp.read(HERE/'vendor/PINNED.json')
    if len(pins)!=11:raise ValueError('Expected eleven frozen numerical sources')
    for rel,expected in pins.items():
        if sp.sha(HERE/'vendor'/rel)!=expected:raise ValueError('Numerical source changed: '+rel)


def port_module():
    spec=importlib.util.spec_from_file_location('disk_compat',HERE/'compat_driver.py')
    port=importlib.util.module_from_spec(spec);sys.modules[spec.name]=port;spec.loader.exec_module(port)
    return port


def require_host_memory():
    import bounded_io as bio
    available=bio.memory_snapshot()['MemAvailable_bytes']
    if available<sp.HOST_START:raise MemoryError('Need 104 GiB available host RAM; found %.2f GiB' % (available/sp.GIB))
    return available


def jobs():
    result=[dict(key='tiny_setup',phase='tiny_setup')]
    result += [dict(key='tiny_reference_'+task,phase='reference',task=task,toy=True) for task in sp.TASKS]
    result.append(dict(key='tiny_assemble',phase='assemble',toy=True))
    result += [dict(key='tiny_build_'+method,phase='build',method=method,toy=True) for method in ('node','direct')]
    result += [dict(key='tiny_check_'+method,phase='tiny_check',method=method,toy=True)
               for method in ('reference','native','node','direct')]
    result += [dict(key='train_'+task,phase='train',task=task) for task in sp.TASKS]
    result += [dict(key='reference_'+task,phase='reference',task=task) for task in sp.TASKS]
    result.append(dict(key='assemble',phase='assemble'))
    result += [dict(key='preflight_'+method,phase='trace',method=method,tag='preflight_'+method)
               for method in ('reference','native')]
    result += [dict(key='build_'+method,phase='build',method=method) for method in ('node','direct')]
    result += [dict(key='admit_'+method,phase='trace',method=method,tag='admit_'+method,admission=True)
               for method in ('node','direct')]
    result += [dict(key=f'r{block}_{method}',phase='trace',method=method,tag=f'r{block}_{method}',block=block)
        for block,order in enumerate(itertools.permutations(('direct','node','native'))) for method in order]
    return result


def identity(root,job):
    return dict(job=job,protocol_sha256=sp.sha(root/'protocol.json'),asset_identity_sha256=sp.sha(root/'asset_identity.json'),
        package_manifest_sha256=sp.sha(HERE/'PACKAGE_MANIFEST.json'))


def phase_done(root,job):
    marker=root/'completed'/(job['key']+'.json')
    if not marker.exists():return False
    record=sp.read(marker)
    if record['identity']!=identity(root,job) or not ds.outputs_match(root,record['outputs']):
        raise ValueError('A completed checkpoint changed: '+job['key'])
    return True


def commit_phase(root,job):
    report_path=root/'phase_reports'/(job['key']+'.json');record=sp.read(report_path)
    if record.get('status')!='COMPLETED' or record['identity']!=identity(root,job) or not ds.outputs_match(root,record['outputs']):
        raise ValueError('Phase output verification failed: '+job['key'])
    outputs=dict(record['outputs']);outputs[str(report_path.relative_to(root))]=sp.sha(report_path)
    sp.dump(root/'completed'/(job['key']+'.json'),dict(identity=record['identity'],outputs=outputs))


def phase_paths(root,job):
    scope=root/'tiny' if job.get('toy') else root;phase=job['phase']
    if phase=='tiny_setup':return [root/'tiny',root/'runtime_contract.json',root/'engine']
    if phase=='train':return [root/'trained/lora'/job['task'],root/'training_logs'/(job['task']+'.jsonl')]
    if phase=='reference':
        names=[job['task']]+(['base'] if job['task']=='mrpc' else [])
        return [scope/'artifacts/references'/(name+suffix) for name in names
                for suffix in ('.safetensors','.safetensors.partial','.json','.json.partial')]
    if phase=='assemble':return [scope/'artifacts/references/metadata.json',scope/'artifacts/references/metadata.json.partial']
    if phase=='build':return [scope/'artifacts/patches'/job['method']]
    if phase=='tiny_check':return [scope/'checks'/(job['method']+'.json'),scope/'checks'/(job['method']+'.json.partial')]
    if phase=='trace':return [root/'stages'/job['tag']]
    raise ValueError('Unknown phase')


def weight_payload(path):
    return '.safetensors' in path.name or path.suffix in ('.bin','.pt','.pth')


def clean_incomplete(root,job):
    """Preserve small diagnostics; discard only this uncommitted phase's payloads."""
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:6]
    destination=root/'interrupted'/(job['key']+'_'+stamp)
    for path in phase_paths(root,job):
        if not path.exists():continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Refusing cleanup outside this run')
        files=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        for file in files:
            if not weight_payload(file):
                target=destination/file.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target)
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()


def launch(root,job,timeout):
    job_file=root/'jobs'/(job['key']+'.json');sp.dump(job_file,job)
    log=root/'logs'/(job['key']+'_'+uuid.uuid4().hex[:6]+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(HERE/'run_8b_disk.py'),'_worker','--result-dir',str(root),'--job',str(job_file)]
    print('PHASE:',job['key'],flush=True);start=time.monotonic()
    with log.open('w') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);offset=0
        try:
            while proc.poll() is None:
                if time.monotonic()-start>timeout:raise subprocess.TimeoutExpired(command,timeout)
                time.sleep(1)
                with log.open() as reader:reader.seek(offset);chunk=reader.read();offset=reader.tell()
                if chunk:print(chunk,end='',flush=True)
            with log.open() as reader:reader.seek(offset);print(reader.read(),end='',flush=True)
        except (KeyboardInterrupt,subprocess.TimeoutExpired):
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait(timeout=5)
            raise
        finally:
            sp.dump(root/'process_exits'/(job['key']+'_'+uuid.uuid4().hex[:6]+'.json'),dict(job=job,pid=proc.pid,
                returncode=proc.returncode,elapsed_seconds=time.monotonic()-start,log=str(log)))
    if proc.returncode:raise RuntimeError('Worker failed: '+str(log)+' (exit '+str(proc.returncode)+')')


def run_job(root,job,timeout):
    if phase_done(root,job):print('REUSE:',job['key'],flush=True);return
    if (root/'phase_reports'/(job['key']+'.json')).exists():
        commit_phase(root,job);print('RECOVER:',job['key'],flush=True);return
    clean_incomplete(root,job);launch(root,job,timeout)
    commit_phase(root,job);print('CHECKPOINT:',job['key'],'VERIFIED',flush=True)


def archive(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and not weight_payload(p)
                 and '__pycache__' not in p.parts and p.name!='manifest.json')
    sp.dump(root/'manifest.json',{str(p.relative_to(root)):sp.sha(p) for p in files})
    output=root.parent/(root.name+'_'+uuid.uuid4().hex[:8]+'.tar.gz')
    with tarfile.open(output,'w:gz',compresslevel=1) as tf:
        for path in files+[root/'manifest.json']:tf.add(path,arcname=root.name+'/'+str(path.relative_to(root)),recursive=False)
    print('SEND_THIS_ARCHIVE:',output,flush=True);print('RESUME_RESULT_DIR:',root,flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','resume','check','_worker'))
    parser.add_argument('--output-dir',type=Path,default=Path('./assets/legacy/ftd_8b_disk_outputs'))
    parser.add_argument('--result-dir',type=Path);parser.add_argument('--job',type=Path)
    parser.add_argument('--phase-timeout',type=int,default=8*3600)
    args=parser.parse_args();verify_package()
    if args.mode=='_worker':
        root=args.result_dir.resolve();job=sp.read(args.job)
        try:
            from disk_engine import worker
            outputs=worker(root,port_module(),job)
            sp.dump(root/'phase_reports'/(job['key']+'.json'),dict(status='COMPLETED',identity=identity(root,job),
                outputs=ds.checkpoint_outputs(root,outputs)))
            return 0
        except Exception as exc:
            sp.dump(root/'worker_failures'/(job['key']+'_'+uuid.uuid4().hex[:6]+'.json'),dict(job=job,
                error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()))
            raise
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=='check':
        try:
            sp.verify_inputs();free=ds.require_space(args.output_dir,sp.START_FREE_DISK);require_host_memory()
            print('CHECK_PASSED: inputs found; free disk %.2f GiB; CUDA has not been tested' % (free/sp.GIB));return 0
        except Exception as exc:print('CHECK_BLOCKED:',str(exc));return 2
    if args.mode=='resume':
        if args.result_dir is None:raise ValueError('resume requires --result-dir')
        root=args.result_dir.resolve()
        if sp.read(root/'release_identity.json')['package_manifest_sha256']!=sp.sha(HERE/'PACKAGE_MANIFEST.json'):
            raise ValueError('Resume requires the same disk release')
    else:
        root=Path(tempfile.mkdtemp(prefix='ftd_8b_disk_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ_'),dir=args.output_dir))
    print('RESULT_DIR:',root,flush=True);ok=False
    def stop(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    try:
        ds.require_space(root,sp.START_FREE_DISK if args.mode=='run' else sp.DISK_FLOOR)
        require_host_memory()
        _,stats,_=sp.verify_inputs();print('FREEZE: computing full source file SHA256',flush=True)
        assets=sp.asset_files(sp.MODEL_DIR,sp.POOL_DIR)
        if args.mode=='resume':
            if assets!=sp.read(root/'asset_identity.json') or sp.read(root/'protocol.json')!=sp.protocol():
                raise ValueError('Frozen source inputs or protocol changed')
        else:
            sp.dump(root/'protocol.json',sp.protocol());sp.dump(root/'asset_identity.json',assets)
            sp.dump(root/'data_audit.json',stats)
            sp.dump(root/'release_identity.json',dict(package_manifest_sha256=sp.sha(HERE/'PACKAGE_MANIFEST.json')))
            shutil.copytree(HERE,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for job in jobs():run_job(root,job,args.phase_timeout)
        from disk_audit import summarize
        summarize(root)
        print('FINAL_ASSET_CHECK:',flush=True)
        if sp.asset_files(sp.MODEL_DIR,sp.POOL_DIR)!=assets:raise ValueError('Source files changed during the run')
        sp.dump(root/'status.json',dict(status='EIGHT_B_DISK_COMPARISON_COMPLETED'))
        print('STATUS: EIGHT_B_DISK_COMPARISON_COMPLETED',flush=True);ok=True
    except (Exception,KeyboardInterrupt) as exc:
        sp.dump(root/'status.json',dict(status='BLOCKED_OR_FAILED',error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()))
        print('Stopped:',str(exc),flush=True)
    finally:archive(root)
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
