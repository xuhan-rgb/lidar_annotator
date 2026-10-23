"""Build a portable source archive, excluding local data, weights and runtimes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TREES = ('tools', 'scripts', 'third_party', 'assets', 'docs', 'requirements')
ROOT_FILES = ('README.md', 'THIRD_PARTY_NOTICES.md', '.gitignore', 'lidar_annotator.py',
              'run.sh', 'run_source.sh', 'test_semantics.py', 'configs/paths.source.json',
              'configs/runtime-id', 'desktop/lidar-annotator.desktop.in',
              'configs/storage.example.json',
              'configs/dependencies.lock.json',
              'models/manifest.json', 'models/sources.json', 'models/download_manifest.json',
              'models/lingbot_model_config.json')
SKIP_DIRS = {'__pycache__', '.git', '.pytest_cache', '.venv', 'node_modules'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.pt', '.pth', '.safetensors', '.so', '.whl', '.log'}
DEMO_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.mp4', '.pdf', '.ipynb'}


def selected_files(root):
    for name in ROOT_FILES:
        yield root / name
    for name in TREES:
        for path in sorted((root / name).rglob('*')):
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts) or path.suffix in SKIP_SUFFIXES:
                continue
            if relative.parts[0] == 'third_party' and path.suffix.lower() in DEMO_SUFFIXES:
                continue
            if path.is_file():
                # Do not publish editable machine-specific profiles from a working tree.
                if relative.parts[:3] == ('tools', 'lidar_annotator', 'configs'):
                    continue
                yield path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def build(root, destination):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    records = {}
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            def add(name, content):
                info = tarfile.TarInfo('lidar-annotator-source/' + name)
                info.size = len(content)
                info.mode = 0o755 if name.endswith('.sh') else 0o644
                archive.addfile(info, io.BytesIO(content))
                records[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            for path in selected_files(root):
                if path.is_symlink():
                    raise ValueError(f'Symlink in source selection; inspect before release: {path}')
                name = path.relative_to(root).as_posix()
                add(name, path.read_bytes())
            manifest = json_bytes(dict(schema='lidar_source_release_v1', files=records))
            info = tarfile.TarInfo('lidar-annotator-source/source_manifest.json')
            info.size = len(manifest); info.mode = 0o644
            archive.addfile(info, io.BytesIO(manifest))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(archive=str(destination), files=len(records), bytes=destination.stat().st_size)


def build_samples(samples, destination):
    """Export raw examples as a separate data artifact, without cached labels."""
    destination=destination.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp');records={}
    try:
        with tarfile.open(temporary,'w:gz') as archive:
            for name in ('examples/kitti_minimal_two_frames','testdata/lmdb_sample'):
                source=samples/name
                if not source.is_dir():raise FileNotFoundError(source)
                for path in sorted(source.rglob('*')):
                    if not path.is_file() or path.is_symlink() or path.suffix in SKIP_SUFFIXES or 'labels' in path.relative_to(source).parts:continue
                    relative=path.relative_to(samples).as_posix();content=path.read_bytes()
                    info=tarfile.TarInfo('lidar-annotator-samples/'+relative);info.size=len(content);info.mode=0o644
                    archive.addfile(info,io.BytesIO(content));records[relative]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
            content=json_bytes(dict(schema='lidar_samples_v1',files=records))
            info=tarfile.TarInfo('lidar-annotator-samples/samples_manifest.json');info.size=len(content);info.mode=0o644
            archive.addfile(info,io.BytesIO(content))
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return dict(archive=str(destination),files=len(records),bytes=destination.stat().st_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/lidar-annotator-source.tar.gz')
    parser.add_argument('--samples', action='store_true', help='export raw samples separately from source code')
    args = parser.parse_args(argv)
    if args.samples:
        import sys
        sys.path.insert(0,str(ROOT))
        from tools.lidar_annotator.storage import directory
        output=args.output if args.output!=ROOT/'dist/lidar-annotator-source.tar.gz' else ROOT/'dist/lidar-annotator-samples.tar.gz'
        result=build_samples(directory('samples'),output)
    else:result=build(ROOT,args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
