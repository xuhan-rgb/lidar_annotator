"""Keep portable data roots separate from annotation workspaces."""
import hashlib,json
from pathlib import Path
from .storage import directory


def work_directory(source,dataset,frames=None):
    source=Path(source).expanduser().resolve();dataset=Path(dataset).expanduser().resolve()
    identity=json.dumps([str(source),str(dataset),sorted(frames) if frames is not None else None])
    token=hashlib.sha256(identity.encode()).hexdigest()[:12]
    return directory('jobs')/(dataset.name+'_'+token)


