"""Dispatch raw sources to one internal image/point/calibration import contract."""
from pathlib import Path


def input_format(source):
    source = Path(source).expanduser().resolve()
    if (source / 'index.lmdb/data.mdb').is_file():return 'lmdb'
    if (source / 'velodyne').is_dir() and any(source.glob('image_*')):return 'kitti'
    raise ValueError('请选择 LMDB 采集根目录，或包含 image_*、velodyne 的 KITTI 序列目录')


def frame_records(source, camera=None):
    if input_format(source) == 'kitti':
        from .kitti_input import frame_records as records
        return records(source, camera=camera)
    from .lmdb_input import frame_records as records
    return records(source)


def read_raw(source, output, frames=None, preview=False, dry_run=False, camera=None, kitti_output=None):
    if input_format(source) == 'kitti':
        from .kitti_input import read_kitti
        return read_kitti(source, output, frames=frames, preview=preview, dry_run=dry_run, camera=camera)
    from .lmdb_input import read_lmdb
    return read_lmdb(source, output, frames=frames, preview=preview, dry_run=dry_run, kitti_output=kitti_output)


def source_image(source, frame, camera=None):
    if input_format(source) == 'kitti':
        from .kitti_input import source_image as image
        return image(source, frame, camera=camera)
    from .sam_prompt_debug import source_image as image
    return image(source, frame)
