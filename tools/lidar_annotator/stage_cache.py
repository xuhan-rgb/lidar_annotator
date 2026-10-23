"""Reuse completed downstream stages only when inputs, code and outputs match."""
import hashlib
import json
from pathlib import Path
from .cli import REPO


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def paths(job,stage):
    source=job/'data/labels/zgc/index.json';person=job/'data/person';vehicle=job/'data/vehicle';temporal=job/'temporal';video=temporal/'map_review/sam3_full_video';centres=video/'interior_half_review/consistent_tracks/centers_only.npz';cars=temporal/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
    inputs={
        'instances':[source,job/'data/visual_instances'],
        'temporal':[source,person/'index.json',vehicle/'index.json',job/'raw/odometry.npz'],
        'tracking':[source,person/'index.json',person/'raw_sam_person',job/'data/actor_video'],
        'optimize':[source,temporal/'poses.npz',temporal/'metadata',temporal/'tracks.json',video/'interior_half_review/map_centers_only.npz',video/'object_category_audit.json',video/'masks'],
        'validate':[source,centres,cars],
        'render':[source,centres,cars,video/'masks'],
        'catalog':[source,centres,cars,job/'review/report.json',person/'metadata',vehicle/'refined'],
    }[stage]
    # Model inference has its own per-frame/per-prompt caches. Include manifests,
    # not enormous immutable checkpoints, in the derived instance-label stage.
    if stage=='instances':inputs += [REPO/'models/manifest.json',REPO/'models/lingbot_model_config.json']
    if stage=='temporal':inputs += [job/'backend.json']
    if stage in ('render','catalog'):
        inputs += [video/'interior_half_review/consistent_tracks/metadata',centres.with_name('final_centers.npz'),video/'native_to_display_ids.json',job/'data/visual_instances',person/'metadata']
    outputs={
        'instances':[person/'frames',person/'metadata',person/'index.json',person/'raw_sam_person',vehicle/'refined',vehicle/'index.json'],
        'temporal':[temporal/'frames',temporal/'metadata',temporal/'poses.npz',temporal/'pose_report.json',temporal/'tracks.json',temporal/'report.json'],
        'tracking':[video/'masks',video/'associations',video/'report.json',video/'visual_presence_index.json',video/'native_to_display_ids.json',video/'interior_half_review/metadata',video/'interior_half_review/map_centers_only.npz',video/'interior_half_review/preparation_report.json'],
        'optimize':[temporal/'map_review/map_points.npz',temporal/'map_review/manifest.json',temporal/'map_review/history_fit',video/'interior_half_review/consistent_tracks'],
        'validate':[job/'validation.json',video/'interior_half_review/consistent_tracks/integrity_audit.json'],
        'render':[job/'review/index.html',job/'review/report.json',job/'review/visualization'],
        'catalog':[job/'labels',job/'annotations.json',job/'label_export.json'],
    }[stage]
    return inputs,outputs


def files(items):
    result=[]
    for item in items:
        if item.is_file():result.append(item)
        elif item.is_dir():result.extend(p for p in item.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return sorted(set(result))


def signature(job,stage):
    inputs,_=paths(job,stage)
    code=[REPO/'tools/lidar_annotator/fresh_pipeline.py',REPO/'tools/lidar_annotator/stage_cache.py']
    code+=list((REPO/'tools/autolabel/zgc_labeling').glob('*.py'))
    if stage=='catalog':code+=[REPO/'tools/lidar_annotator/dataset_labels.py']
    if stage in ('render','catalog'):
        code += [REPO/'tools/lidar_annotator/track_preview.py',REPO/'tools/lidar_annotator/person_support.py',REPO/'tools/lidar_annotator/display_range.py']
    if stage=='instances':code+=[REPO/'tools/autolabel/fresh_instances.py',*list((REPO/'tools/autolabel/pedestrian_instance_zgc').glob('*.py')),*list((REPO/'tools/autolabel/wilddet3d_zgc').glob('*.py'))]
    content=[(str(p),digest(p)) for p in files(inputs+code) if not p.name.startswith('test_') and '/data/actor_video/' not in str(p)]
    # Actor manifests contain an informational reused flag that changes on replay;
    # their stable semantic identity is the ordered source signature instead.
    content=[(name,sha) for name,sha in content if '/data/actor_video/' not in name]
    if stage=='tracking' and (job/'data/actor_video').exists():
        from .actor_video import prompt_source_signature
        for p in sorted((job/'data/actor_video').glob('*/manifest.json')):
            data=json.loads(p.read_text());content.append((str(p),prompt_source_signature(job,data['prompt'])))
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def reuse(job,stage,force=False):
    if force:return False
    receipt=job/'.stage_cache'/f'{stage}.json'
    try:
        data=json.loads(receipt.read_text())
        return data['signature']==signature(job,stage) and bool(data['outputs']) and all((job/name).is_file() and digest(job/name)==sha for name,sha in data['outputs'].items())
    except (OSError,ValueError,KeyError):return False


def complete(job,stage):
    _,outputs=paths(job,stage)
    # Audit appends to a review HTML produced by optimization. It does not change
    # fitted geometry; keep that shared presentation file out of the fit receipt.
    ignore={'index.html','integrity_audit.json','all_ids_review.png'} if stage=='optimize' else set()
    values={str(p.relative_to(job)):digest(p) for p in files(outputs) if p.name not in ignore}
    receipt=job/'.stage_cache'/f'{stage}.json';receipt.parent.mkdir(exist_ok=True)
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(dict(signature=signature(job,stage),outputs=values),indent=2));temp.replace(receipt)
