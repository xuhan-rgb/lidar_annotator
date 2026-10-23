"""Parse explicit original frame IDs and inclusive debug ranges."""
import re


def parse_frames(value):
    if value is None:return None
    text=value if isinstance(value,str) else ' '.join(map(str,value))
    if not text.strip():return None
    result=set()
    for token in re.split(r'[,，\s]+',text.strip()):
        match=re.fullmatch(r'(\d+)(?:-(\d+))?',token)
        if not match:raise ValueError(f'无效帧号或范围：{token}；示例：60-80 125 824')
        lo=int(match[1]);hi=int(match[2]) if match[2] is not None else lo
        if hi<lo:raise ValueError(f'帧范围结束不能小于起始：{token}')
        result.update(range(lo,hi+1))
    return sorted(result)


def debug_output(output,frames):
    """Keep a requested subset separate from an existing full-sequence job."""
    import hashlib,json
    from pathlib import Path
    output=Path(output).expanduser().resolve()
    if frames is not None and (output/'job.json').is_file():
        existing=json.loads((output/'job.json').read_text()).get('frame_ids')
        if existing is not None and existing!=frames:
            token=hashlib.sha256(json.dumps(frames).encode()).hexdigest()[:12]
            return output/'debug'/f'frames_{token}'
    return output


def temporary_debug_output(source,frames):
    """Stable managed debug workspace for a source and explicit original frame IDs."""
    import hashlib,json
    from pathlib import Path
    from .storage import directory
    identity=json.dumps([str(Path(source).expanduser().resolve()),sorted(set(frames))])
    token=hashlib.sha256(identity.encode()).hexdigest()[:20]
    return directory('jobs')/'debug'/token
