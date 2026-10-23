"""Explicit external storage locations; no paths through the source tree."""
import json
import os
from pathlib import Path


def settings():
    config=Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'lidar_annotator/storage.json'
    return json.loads(config.read_text()) if config.is_file() else {}


def data_root():
    default=Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share'))/'lidar_annotator'
    return Path(os.environ.get('LIDAR_DATA_ROOT') or settings().get('data_root') or default).expanduser().resolve()


def directory(name):
    if name not in ('datasets','jobs','samples','experiments','verification','archives'):
        raise ValueError(f'Unknown storage directory: {name}')
    return data_root()/name


def sample_path(name):
    return directory('samples')/name


def default_source(kind):
    if kind not in ('lmdb','kitti'):raise ValueError(kind)
    configured=settings().get(kind+'_source')
    if configured:return Path(configured).expanduser().resolve()
    return sample_path('testdata/lmdb_sample' if kind=='lmdb' else 'examples/kitti_minimal_two_frames')


def default_output(kind):
    if kind not in ('lmdb','kitti'):raise ValueError(kind)
    return directory('datasets' if kind=='lmdb' else 'jobs')/(kind+'_import')


def demo(argv=None):
    from .fresh_pipeline import main
    source=sample_path('examples/kitti_minimal_two_frames')
    if not source.is_dir():raise FileNotFoundError(f'Sample data missing: {source}. Install the separate samples archive first.')
    return main(['--source',str(source),'--output',str(directory('jobs')/'demo'),*(argv or [])])


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=('data_root','datasets','jobs','samples','experiments','verification','archives'))
    args=parser.parse_args(argv)
    print(data_root() if args.name=='data_root' else directory(args.name))


if __name__=='__main__':main()
