"""Remember the latest isolated regeneration without changing older results."""
import json
from pathlib import Path
from uuid import uuid4


def run_root(context):
    base=Path(context['output']).resolve()
    return base.parent/(base.name+'_runs')


def identity(source,frames):return [str(Path(source).expanduser().resolve()),list(frames or [])]


def latest_result(context,source,frames):
    root=run_root(context)
    try:
        record=json.loads((root/'latest.json').read_text())
        result=record['context']
        if record['identity']!=identity(source,frames):return context
        # Only follow one of this task's direct run directories.
        output=Path(result['output']).resolve()
        if output.parent!=root or result['review_output']!=str(output/'review') or result['input_format'] not in ('lmdb','kitti') or result.get('camera')!=context.get('camera'):return context
        return result
    except (OSError,ValueError,KeyError,TypeError):return context


def remember_result(context,source,frames,result):
    root=run_root(context);root.mkdir(parents=True,exist_ok=True)
    temporary=root/('latest_'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(dict(identity=identity(source,frames),context=result),ensure_ascii=False,indent=2))
    temporary.replace(root/'latest.json')
