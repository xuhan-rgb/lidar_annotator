"""Stage wall-clock durations, including failure and cancellation."""
from contextlib import contextmanager
import time


@contextmanager
def measure_stage(entry):
    entry['started_unix']=time.time()
    started=time.perf_counter()
    try:
        yield
    finally:
        entry['elapsed_seconds']=round(max(0.,time.perf_counter()-started),3)
        entry['finished_unix']=time.time()


def stage_timing_text(stages,names,now=None):
    now=time.time() if now is None else now
    rows=[]
    for entry in stages:
        elapsed=entry.get('elapsed_seconds')
        if elapsed is None:
            if entry.get('status')!='running' or 'started_unix' not in entry:continue
            elapsed=max(0.,now-entry['started_unix'])
        minutes,seconds=divmod(max(0.,elapsed),60)
        duration=f'{int(minutes)} 分 {seconds:.1f} 秒' if minutes else f'{seconds:.1f} 秒'
        state={'running':'进行中','complete':'完成','failed':'失败','cancelled':'已停止'}.get(entry.get('status'),entry.get('status',''))
        rows.append(f"{names.get(entry['stage'],entry['stage'])}：{duration}（{state}）")
    return '阶段耗时：\n'+'\n'.join(rows) if rows else '阶段耗时：等待开始'
