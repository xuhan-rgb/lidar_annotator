"""Download and verify the exact official checkpoints used by this pipeline."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verified(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'])


def download(name, source, record, cache):
    target = cache / name
    if verified(target, record):
        print(f'{name}: verified, reuse', flush=True)
        return
    temporary = cache / (name + '.downloading')
    try:
        print(f'{name}: downloading from {source["homepage"]}', flush=True)
        if 'repo_id' in source:
            from huggingface_hub import hf_hub_download
            # Uses HF_TOKEN or the user's hf auth login; never stores credentials here.
            downloaded = hf_hub_download(repo_id=source['repo_id'], filename=source['filename'],
                                         revision=source['revision'])
            shutil.copyfile(downloaded, temporary)
        else:
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'lidar-annotator'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        if not verified(temporary, record):
            raise ValueError(f'Checkpoint size/SHA256 mismatch: {name}; existing target preserved')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'download', 'check'))
    parser.add_argument('--model', choices=('sam3.pt', 'wilddet3d.pt', 'yolo26x.pt'), action='append')
    parser.add_argument('--directory', type=Path,
                        default=Path(os.environ.get('LIDAR_MODEL_CACHE', '~/.cache/lidar_annotator/models')))
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    sources = json.loads((ROOT / 'models/sources.json').read_text())
    selected = args.model or list(manifest)
    cache = args.directory.expanduser().resolve()
    if args.command == 'list':
        print(json.dumps({name: {**manifest[name], **sources[name]} for name in selected}, indent=2))
        return 0
    if args.command == 'check':
        results = {name: verified(cache / name, manifest[name]) for name in selected}
        print(json.dumps(dict(directory=str(cache),verified=results), indent=2))
        return 0 if all(results.values()) else 1
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in selected:
            try:
                download(name, sources[name], manifest[name], cache)
            except Exception as exc:
                # Download errors can include signed URLs. Show a useful, credential-free error.
                print(f'{name}: {type(exc).__name__}. Check access/network/disk space and retry. '
                      f'{sources[name].get("access", sources[name]["homepage"])}', file=sys.stderr)
                return 1
        # Legacy instance inference consumes this official provenance record as well.
        for name in ('manifest.json', 'download_manifest.json'):
            if (cache / name).resolve() != (ROOT / 'models' / name).resolve():
                temporary = cache / (name + '.tmp')
                shutil.copyfile(ROOT / 'models' / name, temporary)
                temporary.replace(cache / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
