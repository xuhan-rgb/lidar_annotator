"""Delete only verified previous jobs for one recording and exact frame selection."""
import fcntl,json,shutil
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4
from .raw_input import frame_records
from .result_runs import run_root
from .frame_history import read_records,write_records


def clear_previous_results(base,current,source,frames,dry_run=False):
    source,records=frame_records(source,camera=base.get('camera'))
    available={int(r['global_frame_id']) for r in records}
    ids=sorted(set(frames)) if frames is not None else sorted(available)
    if not ids or set(ids)-available:raise ValueError('指定帧号不在采集索引中')
    root=run_root(base)
    candidates=[Path(base['output']),Path(current['output'])]
    if root.is_dir():candidates.extend(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    owned=[]
    for path in dict.fromkeys(candidates):
        if not path.exists():continue
        if path.is_symlink():raise ValueError(f'不能清理符号链接任务目录：{path}')
        path=path.resolve();marker=path/'job.json'
        if path==source or path in source.parents:raise ValueError('不能清理原始采集目录')
        if not marker.is_file():continue
        job=json.loads(marker.read_text())
        if job.get('schema')=='lidar_fresh_job_v1' and Path(job['source']).resolve()==source and sorted(job['frame_ids'])==ids and job.get('camera')==base.get('camera'):
            owned.append(path)
    if dry_run:return owned
    # Check every lock before deleting anything, including jobs in another window.
    with ExitStack() as locks:
        for path in owned:
            lock=locks.enter_context((path/'.run.lock').open('a'))
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError(f'任务仍在运行，不能清理：{path}') from None
        for path in owned:
            assets=[]
            if (path/'kitti/labels').is_dir():assets.append(path/'kitti')
            if (path/'label_versions').is_dir():assets.append(path/'label_versions')
            if assets:
                archive=root/'label_archives'/uuid4().hex;archive.mkdir(parents=True)
                for asset in assets:shutil.move(str(asset),archive/asset.name)
                for name in ('job.json','recipe.json','label_export.json'):
                    if (path/name).is_file():shutil.copy2(path/name,archive/name)
                print('已保留标签与历史版本：'+str(archive),flush=True)
            shutil.rmtree(path)
        pointer=root/'latest.json'
        if pointer.is_file():
            try:previous=Path(json.loads(pointer.read_text())['context']['output']).resolve()
            except (ValueError,KeyError,TypeError):previous=None
            if previous in owned:pointer.unlink()
        entries=read_records();remaining=[e for e in entries if Path(e['context']['output']).resolve() not in owned]
        if remaining!=entries:write_records(remaining)
    return owned
