"""Atomic per-stage progress for Qt, with one updating line in a terminal."""
import json
import os
from pathlib import Path
import sys
import time


def report_progress(stage,completed,total,frame=None,detail='',job=None):
    completed,total=int(completed),int(total)
    if total<0 or not 0<=completed<=total:raise ValueError('Invalid progress counters')
    data=dict(stage=stage,completed=completed,total=total,frame=frame,detail=detail,updated_unix=time.time(),pid=os.getpid())
    output=job or os.environ.get('LIDAR_JOB_OUTPUT')
    if output:
        path=Path(output)/'progress.json';path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(f'.progress.{os.getpid()}.tmp');temp.write_text(json.dumps(data,ensure_ascii=False));temp.replace(path)
    if sys.stderr.isatty():
        fraction=completed/total if total else 0
        text=f'\r{stage}: [{"="*int(30*fraction):<30}] {completed}/{total} {fraction:.1%}'
        if frame is not None:text+=f' frame={frame}'
        if detail:text+=f' {detail}'
        sys.stderr.write(text+'\033[K'+ ('\n' if total and completed==total else ''));sys.stderr.flush()
    return data
