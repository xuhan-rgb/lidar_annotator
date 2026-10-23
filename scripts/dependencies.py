"""Fetch exact GitHub source commits, project patches and official font assets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.autolabel.resource_paths import LOCK_PATH,dependency_root


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:raise ValueError(f'Unsafe dependency path: {value}')
    return Path(*path.parts)


def download(spec,downloads):
    destination=downloads/spec['sha256']
    if destination.is_file() and digest(destination)==spec['sha256']:return destination
    temporary=destination.with_suffix('.downloading')
    try:
        request=urllib.request.Request(spec['url'],headers={'User-Agent':'lidar-annotator-dependencies'})
        with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as stream:
            shutil.copyfileobj(response,stream,1024*1024)
        if digest(temporary)!=spec['sha256']:raise ValueError(f'Download SHA256 mismatch: {spec["name"]}')
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return destination


def extract_sources(archive_path,destination,include):
    allowed=[PurePosixPath(name) for name in include]
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            relative(member.name)
            if len(path.parts)<2:continue
            name=PurePosixPath(*path.parts[1:])
            if not any(name==p or p in name.parents for p in allowed):continue
            if member.isdir():continue
            if not member.isfile():raise ValueError(f'Unsupported source archive entry: {member.name}')
            output=destination/relative(str(name));output.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,output.open('wb') as target:shutil.copyfileobj(source,target)
    for name in allowed:
        if not (destination/relative(str(name))).exists():raise FileNotFoundError(f'Missing archive selection: {name}')


def apply_patches(source,spec,project):
    for item in spec.get('patches',[]):
        patch=project/relative(item['file'])
        if digest(patch)!=item['sha256']:raise ValueError(f'Project patch SHA256 mismatch: {patch}')
        for change in item['files']:
            if digest(source/relative(change['file']))!=change['upstream_sha256']:
                raise ValueError(f'Patch base differs from pinned commit: {change["file"]}')
        subprocess.run(['git','apply','--check',str(patch.resolve())],cwd=source,check=True,capture_output=True,text=True)
        subprocess.run(['git','apply',str(patch.resolve())],cwd=source,check=True,capture_output=True,text=True)
        for change in item['files']:
            if digest(source/relative(change['file']))!=change['patched_sha256']:
                raise ValueError(f'Patched source SHA256 mismatch: {change["file"]}')


def read_lock(path):
    data=json.loads(Path(path).read_text())
    if data['schema']!='lidar_remote_dependencies_v1':raise ValueError('Unsupported dependency lock schema')
    for spec in [*data['sources'],*data['assets']]:
        if not re.fullmatch('[0-9a-f]{40}',spec['commit']):raise ValueError(f'Full commit required: {spec["name"]}')
        if not re.fullmatch('[0-9a-f]{64}',spec['sha256']):raise ValueError(f'SHA256 required: {spec["name"]}')
    return data


def check(lock_path=LOCK_PATH,cache_base=None,project=ROOT):
    root=dependency_root(lock_path,cache_base)
    try:
        lock=read_lock(lock_path);receipt=json.loads((root/'ready.json').read_text())
        if receipt['lock_sha256']!=digest(lock_path):return False
        for source in lock['sources']:
            for patch in source.get('patches',[]):
                if digest(project/relative(patch['file']))!=patch['sha256']:return False
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'ready.json' and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
        if actual!=set(receipt['files']):return False
        return bool(receipt['files']) and all((root/relative(name)).is_file() and digest(root/relative(name))==sha for name,sha in receipt['files'].items())
    except (OSError,ValueError,KeyError,TypeError):return False


def fetch(lock_path=LOCK_PATH,cache_base=None,project=ROOT):
    lock=read_lock(lock_path);root=dependency_root(lock_path,cache_base)
    for source in lock['sources']:
        for patch in source.get('patches',[]):
            if digest(project/relative(patch['file']))!=patch['sha256']:
                raise ValueError(f'Project patch SHA256 mismatch: {patch["file"]}')
    root.parent.mkdir(parents=True,exist_ok=True);downloads=root.parent/'downloads';downloads.mkdir(exist_ok=True)
    with (root.parent/'.fetch.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if check(lock_path,cache_base,project):print('Locked dependencies verified; reuse '+str(root),flush=True);return root
        # Never replace a modified cache silently. A lock change creates a new root.
        if root.exists():raise ValueError(f'Dependency cache is incomplete or modified: {root}; inspect/remove it and fetch again')
        if any(source.get('patches') for source in lock['sources']) and not shutil.which('git'):
            raise RuntimeError('Git is required to apply the locked project patches. Install git before fetching dependencies.')
        with tempfile.TemporaryDirectory(prefix='dependency-install-',dir=root.parent) as temp:
            stage=Path(temp)/'sources';stage.mkdir()
            for spec in lock['sources']:
                print(f"Preparing {spec['name']} @ {spec['commit']}",flush=True)
                target=stage/relative(spec['target']);target.mkdir(parents=True,exist_ok=True)
                extract_sources(download(spec,downloads),target,spec['include'])
                apply_patches(target,spec,project)
            for spec in lock['assets']:
                print('Preparing font asset '+spec['name'],flush=True)
                archive=download(spec,downloads)
                if 'members' in spec:
                    with tarfile.open(archive) as bundle:
                        for member,name in spec['members'].items():
                            entry=bundle.getmember(member)
                            if not entry.isfile():raise ValueError(f'Unsupported font entry: {member}')
                            target=stage/relative(name);target.parent.mkdir(parents=True,exist_ok=True)
                            with bundle.extractfile(entry) as source,target.open('wb') as stream:shutil.copyfileobj(source,stream)
                else:
                    target=stage/relative(spec['target']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive,target)
            values={p.relative_to(stage).as_posix():digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
            commits={source['name']:dict(repository=source['repository'],commit=source['commit']) for source in lock['sources']}
            (stage/'ready.json').write_text(json.dumps(dict(lock_sha256=digest(lock_path),sources=commits,files=values),indent=2)+'\n')
            stage.rename(root)
        return root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('list','fetch','check','path'))
    parser.add_argument('--quiet',action='store_true');args=parser.parse_args(argv)
    if args.command=='path':print(dependency_root());return 0
    if args.command=='list':print(LOCK_PATH.read_text());return 0
    if args.command=='fetch':fetch();return 0
    valid=check()
    if not args.quiet:print(json.dumps(dict(root=str(dependency_root()),verified=valid),indent=2))
    return 0 if valid else 1


if __name__=='__main__':raise SystemExit(main())
