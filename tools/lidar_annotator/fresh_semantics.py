"""Generate fresh ZGC semantic masks from raw LMDB-imported frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .class_semantic_cache import save_class_cache, load_class_cache

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from autolabel.task_progress import report_progress
from autolabel.runtime_paths import MODEL_CACHE  # noqa: E402
from autolabel.semantic_zgc.centerpoint_weak_semantics import weak_targets  # noqa: E402

REPO = HERE.parents[1]
RECIPE_PATH = REPO / 'tools/autolabel/zgc_mask_recipe_v1/recipe.json'
RAW_CLASSES = 10
UNKNOWN = 255
# Progress reporting leaves the semantic algorithm unchanged; accept its exact prior signature.
PRE_PROGRESS_RECIPE_SHA256 = 'e40abd8c86dfa518a841647d72ce491783c6e5c19559f7f37d3852237f8f5244'
PRE_PROGRESS_SIGNATURE = '3cdeee3d7b090719d713cb76cdbaeec2b4e5dabf13a8e4f7dd8ad7eb319c6ee8'
PROMPT_CACHE_POLICY = 'sam3_prompt_evidence_v1'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _prompt_cache_path(job, source_sha, model_sha, prompt, threshold, recipe, local_id):
    policy = dict(version=PROMPT_CACHE_POLICY, source_sha=source_sha, model_sha=model_sha,
                  prompt=prompt, candidate_threshold=recipe['model']['instance_candidate_threshold'],
                  score_threshold=threshold, mask_threshold=recipe['base_semantics']['mask_threshold'],
                  inference=recipe['model'].get('inference_autocast', 'cuda bfloat16'),
                  policy_version=PROMPT_CACHE_POLICY)
    key = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    return Path(job) / 'data/prompt_cache' / key / f'{local_id:06d}.npz'


def _save_prompt_cache(path, evidence, strong, scores, masks, records):
    ys, xs = np.where((evidence > 0) | (strong > 0))
    if len(xs):
        box = np.asarray([xs.min(), ys.min(), xs.max()+1, ys.max()+1], np.int32)
        crop = evidence[box[1]:box[3], box[0]:box[2]].astype(np.float32)
        strong_crop = strong[box[1]:box[3], box[0]:box[2]].astype(np.float32)
    else:
        box = np.zeros(4, np.int32); crop = np.zeros((0, 0), np.float32); strong_crop = crop.copy()
    h, w = evidence.shape
    packed = np.packbits(np.asarray(masks, bool).reshape(-1, h*w), axis=1, bitorder='little') if masks else np.zeros((0,(h*w+7)//8), np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, shape=np.asarray([h,w],np.int32), bbox=box, evidence=crop,
                        strong=strong_crop, scores=np.asarray(scores,np.float32), masks=packed,
                        records=np.asarray(json.dumps(records)))
    tmp.replace(path)
    _json(path.with_suffix('.json'), dict(output_sha256=_sha(path)))


def _load_prompt_cache(path, shape):
    try:
        manifest=json.loads(path.with_suffix('.json').read_text())
        if manifest.get('output_sha256') != _sha(path): return None
        with np.load(path, allow_pickle=False) as data:
            if tuple(data['shape']) != tuple(shape): return None
            h,w=shape; box=data['bbox']; ev=np.zeros(shape,np.float32); strong=np.zeros(shape,np.float32)
            if box[2] > box[0] and box[3] > box[1]:
                ev[box[1]:box[3],box[0]:box[2]]=data['evidence']
                strong[box[1]:box[3],box[0]:box[2]]=data['strong']
            packed=data['masks']; masks=np.unpackbits(packed,axis=1,count=h*w,bitorder='little').reshape(-1,h,w).astype(bool)
            return ev,strong,data['scores'].tolist(),masks,json.loads(str(data['records'].item()))
    except (OSError, ValueError, KeyError, EOFError):
        return None


def cached_prompt_result(path, shape, force, infer):
    """Reuse a valid prompt artifact, otherwise run infer once and persist it."""
    result = None if force else _load_prompt_cache(path, shape) if path.is_file() else None
    if result is None:
        result = infer()
        _save_prompt_cache(path, *result)
    return result


def _semantic_prompts(recipe):
    """Return prompts and their semantic class, excluding disabled classes."""
    disabled=set(recipe.get('disabled_classes', [])); base=recipe['base_semantics']['prompts']
    class_prompts={int(cid):list(prompts) for cid,prompts in base.items()}
    class_prompts[0] += list(recipe['base_semantics']['grass_prompts'])
    class_prompts[0] += list(recipe['ground_refinement']['prompts'])
    class_prompts[0] += [recipe['manhole_covers']['prompt'],recipe['small_debris']['prompt']]
    class_prompts[8]=[recipe['ground_objects']['prompt']]
    class_prompts[9]=[recipe['roof_union']['roof_prompt'],*recipe['roof_union']['canopy_prompts']]
    return {cid:prompts for cid,prompts in class_prompts.items() if cid not in disabled}



def _prompt_thresholds(recipe):
    prompts_by_class = _semantic_prompts(recipe)
    thresholds = {p: recipe['base_semantics']['score_threshold']
                  for ps in prompts_by_class.values() for p in ps}
    if 8 in prompts_by_class:
        thresholds[recipe['ground_objects']['prompt']] = recipe['ground_objects']['minimum_instance_score']
    if 0 in prompts_by_class:
        thresholds.update(dict.fromkeys(recipe['ground_refinement']['prompts'], recipe['ground_refinement']['minimum_instance_score']))
        thresholds[recipe['manhole_covers']['prompt']] = recipe['manhole_covers']['minimum_instance_score']
        thresholds[recipe['small_debris']['prompt']] = recipe['small_debris']['minimum_instance_score']
    if 9 in prompts_by_class:
        thresholds[recipe['roof_union']['roof_prompt']] = recipe['roof_union']['roof_threshold']
        thresholds.update(dict.fromkeys(recipe['roof_union']['canopy_prompts'], recipe['roof_union']['canopy_threshold']))
    return thresholds


def _class_cache_path(job, source_sha, model_sha, cid, recipe, thresholds, local_id):
    prompts = _semantic_prompts(recipe)[cid]
    policy = dict(version='sam3_class_evidence_v1', source=source_sha, model=model_sha, class_id=cid,
                  prompts=[(p, thresholds.get(p, recipe['base_semantics']['score_threshold'])) for p in prompts],
                  base_prompts=recipe['base_semantics']['prompts'].get(str(cid), []),
                  candidate_threshold=recipe['model']['instance_candidate_threshold'],
                  mask_threshold=recipe['base_semantics']['mask_threshold'],
                  inference=recipe['model'].get('inference_autocast', 'cuda bfloat16'))
    if cid == 0:
        policy['ground'] = {k: recipe[k] for k in ('ground_refinement', 'manhole_covers', 'small_debris')}
        policy['grass'] = recipe['base_semantics']['grass_prompts']
    if cid == 9: policy['roof'] = recipe['roof_union']
    key = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    return Path(job) / 'data/class_cache' / str(cid) / f'{local_id:06d}_{key}.npz'


def _aggregate_class(cid, evidence, actor_strong, recipe):
    """Retain only the exact sufficient statistics consumed by composition."""
    def maximum(prompts):
        return np.maximum.reduce([evidence[p] for p in prompts])
    if cid < 8:
        planes = dict(base=maximum(recipe['base_semantics']['prompts'][str(cid)]))
    elif cid == 8:
        planes = dict(base=evidence[recipe['ground_objects']['prompt']])
    else:
        roof = recipe['roof_union']; canopy = [evidence[p] for p in roof['canopy_prompts']]
        votes = sum(x > 0 for x in canopy)
        planes = dict(base=np.maximum(evidence[roof['roof_prompt']],
            np.where(votes >= roof['minimum_canopy_pixel_support'], np.maximum.reduce(canopy), 0)))
    if cid in (1, 2): planes['strong'] = actor_strong[cid - 1]
    if cid == 0:
        ref = recipe['ground_refinement']; values = [evidence[p] for p in ref['prompts']]
        planes['grass'] = maximum(recipe['base_semantics']['grass_prompts'])
        planes['pavement'] = np.where(sum(x > 0 for x in values) >= ref['minimum_pixel_prompt_support'], np.maximum.reduce(values), 0)
        planes['cover'] = evidence[recipe['manhole_covers']['prompt']]
        planes['debris'] = evidence[recipe['small_debris']['prompt']]
    return planes


def _expand_classes(classes, recipe, shape):
    """Adapt class summaries to the unchanged production composer."""
    zero = np.zeros(shape, np.float32)
    evidence = {p: zero for ps in _semantic_prompts({**recipe, 'disabled_classes': []}).values() for p in ps}
    strong = np.zeros((2, *shape), np.float32)
    for cid, planes in classes.items():
        if cid < 8:
            for prompt in recipe['base_semantics']['prompts'][str(cid)]: evidence[prompt] = planes['base']
        elif cid == 8:
            evidence[recipe['ground_objects']['prompt']] = planes['base']
        else:
            evidence[recipe['roof_union']['roof_prompt']] = planes['base']
        if cid in (1, 2): strong[cid - 1] = planes['strong']
        if cid == 0:
            for prompt in recipe['base_semantics']['grass_prompts']: evidence[prompt] = planes['grass']
            for prompt in recipe['ground_refinement']['prompts']: evidence[prompt] = planes['pavement']
            evidence[recipe['manhole_covers']['prompt']] = planes['cover']
            evidence[recipe['small_debris']['prompt']] = planes['debris']
    return evidence, strong


def _actor_manifest_signature(source_signatures,prompts):
    h=hashlib.sha256()
    for prompt in prompts:
        h.update(prompt.encode())
        h.update(str(source_signatures[prompt]).encode())
    return h.hexdigest()


def zbuffer_visible(pixels_uv, camera_depth, in_image, shape):
    """Return nearest-return visibility; FOV membership remains a separate input."""
    h, w = shape
    uv = np.asarray(pixels_uv)
    depth = np.asarray(camera_depth)
    fov = np.asarray(in_image, bool)
    finite = np.isfinite(uv).all(1) & np.isfinite(depth) & (depth > 0)
    xy = np.zeros(uv.shape, np.int64)
    xy[finite] = np.floor(uv[finite]).astype(np.int64)
    inside = finite & fov & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = np.flatnonzero(inside)
    nearest = np.full(h * w, np.inf, np.float32)
    flat = xy[ids, 1] * w + xy[ids, 0]
    np.minimum.at(nearest, flat, depth[ids])
    visible = np.zeros(len(uv), bool)
    visible[ids] = depth[ids] <= nearest[flat] + np.maximum(1e-3, nearest[flat] * 1e-4)
    return visible


def project_labels(arrays, labels, confidence):
    uv = np.asarray(arrays['pixels_uv']); visible = np.asarray(arrays['visible'], bool)
    h, w = labels.shape
    finite = np.isfinite(uv).all(1)
    xy = np.zeros(uv.shape, np.int64); xy[finite] = np.floor(uv[finite]).astype(np.int64)
    keep = visible & finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out = np.full(len(uv), UNKNOWN, np.uint8); score = np.zeros(len(uv), np.float32)
    ii = np.flatnonzero(keep); q = xy[ii]
    out[ii] = labels[q[:, 1], q[:, 0]]; score[ii] = confidence[q[:, 1], q[:, 0]]
    score[out == UNKNOWN] = 0
    return out, score


def training_labels(labels):
    result = np.asarray(labels).copy()
    result[np.isin(result, (7, 8))] = 6
    result[result == 9] = 4
    return result


def confidence_for_labels(scores, labels):
    """Select confidence from the score plane matching each final base class."""
    confidence = np.take_along_axis(scores, np.minimum(labels, 7)[None], axis=0)[0].copy()
    confidence[labels == UNKNOWN] = 0
    return confidence


def teacher_bev_from_occupancy_bits(occupancy_bits):
    """Convert network [Y,X] occupancy bits to source [X,Y] teacher labels."""
    teacher = np.full(occupancy_bits.shape, UNKNOWN, np.uint8)
    for label, occupancy in enumerate((0, 1, 2, 5, 4, 6, 3)):
        teacher[occupancy_bits == (1 << occupancy)] = label
    return teacher.T.copy()


def apply_ground_refinements(labels, confidence, evidence, recipe, include_ground_objects=True,
                             include_refinements=True):
    """Apply confirmed ground-object, pavement, cover, and debris policies."""
    out, conf = labels.copy(), confidence.copy()
    if include_ground_objects:
        obj = recipe['ground_objects']; ev = evidence[obj['prompt']]
        chosen = (ev > 0) & np.isin(out, obj['replacement_classes'])
        out[chosen] = obj['output_class']; conf[chosen] = ev[chosen]
    if not include_refinements: return out,conf
    ref = recipe['ground_refinement']; prompts = ref['prompts']
    votes = sum(evidence[p] > 0 for p in prompts)
    pavement_score = np.maximum.reduce([evidence[p] for p in prompts])
    selected = (votes >= ref['minimum_pixel_prompt_support']) & np.isin(out, ref['replacement_classes'])
    out[selected] = 0; conf[selected] = np.maximum(conf[selected], pavement_score[selected])
    cover = recipe['manhole_covers']; ev = evidence[cover['prompt']]
    selected = (ev > 0) & np.isin(out, cover['replacement_classes'])
    out[selected] = cover['output_class']; conf[selected] = np.maximum(conf[selected], ev[selected])
    debris = recipe['small_debris']; ev = evidence[debris['prompt']]
    count, components = cv2.connectedComponents((out == debris['existing_object_class']).astype(np.uint8))
    sizes = np.bincount(components.ravel(), minlength=count)
    support = ev > 0
    overlaps = np.bincount(components[support], minlength=count)
    clear = overlaps >= debris['minimum_connected_component_overlap'] * sizes
    clear[0] = False
    selected = clear[components]
    out[selected] = debris['output_class']
    comp_score = np.zeros(count, np.float32)
    np.maximum.at(comp_score, components[support], ev[support])
    conf[selected] = np.maximum(conf[selected], comp_score[components[selected]])
    return out, conf


def apply_roof_union(labels, confidence, evidence, recipe):
    roof = recipe['roof_union']; roof_score = evidence[roof['roof_prompt']]
    canopy = [evidence[p] for p in roof['canopy_prompts']]
    votes = sum(x > 0 for x in canopy)
    canopy_score = np.maximum.reduce(canopy)
    selected = (labels == roof['replacement_class']) & (
        (roof_score > 0) | (votes >= roof['minimum_canopy_pixel_support']))
    chosen_score = np.maximum(roof_score, np.where(votes >= roof['minimum_canopy_pixel_support'], canopy_score, 0))
    out, conf = labels.copy(), confidence.copy()
    out[selected] = roof['output_class']; conf[selected] = chosen_score[selected]
    return out, conf


def _render_overlay(image, labels):
    colors = np.array([[55,190,65],[0,255,255],[255,220,0],[65,135,240],
                       [160,105,225],[255,115,15],[245,65,60],[255,80,160],
                       [245,65,60],[160,105,225]], np.uint8)
    rgb = np.asarray(image).copy(); known = labels != UNKNOWN
    rgb[known] = (rgb[known] * .45 + colors[labels[known]] * .55).astype(np.uint8)
    return Image.fromarray(rgb)


def compose_image_semantics(evidence, actor_strong, recipe):
    """Compose image-only prompt evidence using the production semantic rules."""
    prompts_by_class=_semantic_prompts(recipe)
    actor_prompts=prompts_by_class.get(1,[])+prompts_by_class.get(2,[])
    shape=actor_strong.shape[1:]
    scores=np.zeros((8,*shape),np.float32)
    for cls, prompts in recipe['base_semantics']['prompts'].items():
        if int(cls) in recipe.get('disabled_classes',[]): continue
        for prompt in prompts: scores[int(cls)]=np.maximum(scores[int(cls)],evidence[prompt])
    class_scores=scores.copy()
    labels=class_scores.argmax(0).astype(np.uint8); confidence=class_scores.max(0); labels[confidence==0]=UNKNOWN
    from .composition_rules import candidate_scores, apply_priorities, apply_subtractions
    actor_support=np.maximum(actor_strong[0],actor_strong[1])
    unsupported=np.isin(labels,(1,2)) & (actor_support==0); labels[unsupported]=UNKNOWN
    confidence=confidence_for_labels(scores,labels)
    candidates=candidate_scores(evidence,scores,actor_strong,recipe)
    labels,confidence=apply_priorities(labels,confidence,scores,candidates,actor_strong,evidence,recipe)
    # Fresh predictions compete by score; old building/fence conflict handling
    # depended on previous teacher labels, which are intentionally untrusted here.
    if 0 in prompts_by_class or 8 in prompts_by_class:
        labels,confidence=apply_ground_refinements(labels,confidence,evidence,recipe,
            include_ground_objects=8 in prompts_by_class,include_refinements=0 in prompts_by_class)
    labels,confidence=apply_subtractions(labels,confidence,candidates,recipe)
    # Ignore unconfirmed actor pixels, while retaining SAM detections for instance review.
    actors=np.maximum.reduce([evidence[p] for p in actor_prompts]) if actor_prompts else np.zeros(shape,np.float32)
    actor_area=np.isin(labels,(1,2)) & (actors==0); labels[actor_area]=UNKNOWN; confidence[actor_area]=0
    confidence[labels==UNKNOWN]=0
    return labels,confidence


def _run_model(rows, raw_root, job, refresh, recipe, refresh_prompts=None, recipe_path=None):
    weight = MODEL_CACHE / 'sam3.pt'
    expected_weight = recipe['model'].get('sha256', recipe['model'].get('weight_sha256'))
    model_sha = expected_weight or (_sha(weight) if weight.is_file() else 'missing')
    sig = hashlib.sha256()
    recipe_path = Path(recipe_path or RECIPE_PATH)
    for path in (recipe_path, Path(__file__), HERE / 'class_semantic_cache.py'):
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sig.update(block)
    signature = sig.hexdigest()
    backend_path=job/'backend.json'
    backend=json.loads(backend_path.read_text()) if backend_path.is_file() else {}
    actor_video=backend.get('actor_mask_source')=='sam3_video'
    actor_reader=None
    actor_signature_reader=None
    if actor_video:
        from tools.lidar_annotator.actor_video import read_prompt as actor_read_prompt
        from tools.lidar_annotator.actor_video import prompt_source_signature
        actor_reader=actor_read_prompt
        actor_signature_reader=prompt_source_signature
    root = job / 'data/labels/zgc'
    for folder in ('image_masks', 'confidence', 'overlay', 'metadata', 'frames'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (job / 'data/visual_instances').mkdir(parents=True, exist_ok=True)
    prompts_by_class=_semantic_prompts(recipe)
    prompt_class={prompt:cid for cid,ps in prompts_by_class.items() for prompt in ps}
    person_prompts=prompts_by_class.get(1,[]); vehicle_prompts=prompts_by_class.get(2,[])
    actor_prompts=person_prompts+vehicle_prompts
    all_prompts=[p for cid,ps in prompts_by_class.items() for p in ps
                 if not (actor_video and cid in (1,2))]
    thresholds = _prompt_thresholds(recipe)
    prompt_desc=[]
    for cid,ps in prompts_by_class.items():
        name=next((c['name'] for c in recipe.get('raw_classes',[]) if c['id']==cid),str(cid))
        prompt_desc.append(f"class={cid} {name}: "+', '.join(f'{p!r}@{thresholds[p]:g}' for p in ps))
    if actor_video:
        prompt_desc.append('actor provider=sam3_video; prompts: '+', '.join(actor_prompts))
    print('semantic prompts: ' + '; '.join(prompt_desc), flush=True)
    actor_source_signatures={p:actor_signature_reader(job,p,recipe) for p in actor_prompts} if actor_video else {}
    if actor_video and any(s is None for s in actor_source_signatures.values()):
        raise RuntimeError('Actor video cache is missing or invalid for one or more active prompts')
    if not actor_video: actor_source_signatures=None
    actor_manifest_signature=_actor_manifest_signature(actor_source_signatures,actor_prompts) if actor_video else None
    if actor_video:
        for prompt in actor_prompts:
            source_sig=actor_source_signatures[prompt]
            cid=1 if prompt in person_prompts else 2
            name=next((c['name'] for c in recipe.get('raw_classes',[]) if c['id']==cid),str(cid))
            print(f"semantic actor cache class={cid} {name} prompt={prompt!r} source={source_sig}",flush=True)
    migration_path = job / 'data/class_cache/migration.json'
    migration = json.loads(migration_path.read_text()) if migration_path.is_file() else {}
    cache_counts={p:[0,0] for p in all_prompts}
    model = processor = None
    rows_out = []
    for local_id, row in enumerate(rows):
        actor_results={}; actor_signature=actor_manifest_signature
        stem = f'{local_id:06d}'; mask_path=root/'image_masks'/f'{stem}.png'; conf_path=root/'confidence'/f'{stem}.npz'
        overlay_path=root/'overlay'/f'{stem}.jpg'; meta_path=root/'metadata'/f'{stem}.json'
        output_paths=[mask_path,conf_path,overlay_path,meta_path,root/'frames'/f'{stem}.npz',job/'data/visual_instances'/f'{stem}.npz',job/'data/visual_instances'/f'{stem}.json']
        if not refresh and meta_path.is_file():
            meta=json.loads(meta_path.read_text())
            stored_meta_hash=meta.pop('metadata_sha256', None)
            computed_meta_hash=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            meta['metadata_sha256']=stored_meta_hash
            outputs_valid=all(p.is_file() and _sha(p)==meta.get('output_sha256',{}).get(str(p.relative_to(job))) for p in output_paths if p != meta_path)
            if (stored_meta_hash==computed_meta_hash and meta.get('source_payload_sha256')==row['original_payload_sha256']
                    and (meta.get('signature')==signature or
                         migration.get('signature')==signature and migration.get('metadata_sha256',{}).get(stem)==stored_meta_hash or
                         meta.get('signature')==PRE_PROGRESS_SIGNATURE and _sha(recipe_path)==PRE_PROGRESS_RECIPE_SHA256) and outputs_valid):
                if meta.get('actor_provider_signatures')==actor_source_signatures:
                    rows_out.append(meta['row'])
                    report_progress('semantic',local_id+1,len(rows),frame=row['frame_id'],detail='cached',job=job)
                    continue
        if actor_video:
            for prompt in actor_prompts:
                actor_results[prompt]=actor_reader(job,prompt,local_id,recipe)
        source_npz=raw_root/row['path']
        if _sha(source_npz) != row.get('npz_sha256', _sha(source_npz)): raise ValueError(f'Raw frame hash mismatch: {source_npz}')
        arrays={k:v for k,v in np.load(source_npz,allow_pickle=False).items()}
        image=Image.open(row['image_path']).convert('RGB'); shape=(image.height,image.width)
        arrays['visible']=zbuffer_visible(arrays['pixels_uv'], arrays['camera_depth'], arrays['in_image'], shape)
        all_configured_prompts=_semantic_prompts({**recipe,'disabled_classes':[]})
        evidence={p:np.zeros(shape,np.float32) for ps in all_configured_prompts.values() for p in ps}
        prompt_records={}; visual_masks=[]; visual_records=[]; actor_strong=np.zeros((2,*shape),np.float32); encoder_state=None
        if actor_video:
            for prompt in actor_prompts:
                result=actor_results[prompt]
                ev=np.asarray(result['evidence'],np.float32); strong=np.asarray(result['strong'],np.float32)
                if ev.shape!=shape or strong.shape!=shape: raise ValueError(f'Actor provider shape mismatch for {prompt!r}')
                masks=np.asarray(result['masks'],bool)
                if masks.ndim!=3 or tuple(masks.shape[1:])!=shape: raise ValueError(f'Actor provider mask shape mismatch for {prompt!r}')
                evidence[prompt]=ev; prompt_records[prompt]=list(result['scores'])
                target=0 if prompt in person_prompts else 1
                actor_strong[target]=np.maximum(actor_strong[target],strong)
                visual_masks.extend(masks)
                visual_records.extend(result['records'])
        from contextlib import nullcontext
        class_results = {}; class_paths = {}; class_instances = {}
        video_masks = list(visual_masks); video_records = list(visual_records)
        for cid, prompts in prompts_by_class.items():
            if actor_video and cid in (1, 2): continue
            path = _class_cache_path(job, row['original_payload_sha256'], model_sha, cid, recipe, thresholds, local_id)
            class_paths[cid] = path
            force = bool(refresh and (refresh_prompts is None or any(p in refresh_prompts for p in prompts)))
            cached = None if force else load_class_cache(path, shape)
            if cached is None: continue
            planes, details = cached
            class_results[cid] = planes
            prompt_records.update(details['prompt_scores'])
            class_instances[cid] = ([planes[k] for k in details['instance_planes']], details['records'])
        expanded, cached_strong = _expand_classes(class_results, recipe, shape)
        for cid in class_results:
            for prompt in prompts_by_class[cid]: evidence[prompt] = expanded[prompt]
            if cid in (1, 2): actor_strong[cid - 1] = cached_strong[cid - 1]
        cache_entries=[]; pending_prompts=[]
        for prompt in all_prompts:
            cid = prompt_class[prompt]
            if cid in class_results:
                cache_counts[prompt][0] += 1
                continue
            pending_prompts.append(prompt)
            cache_path=_prompt_cache_path(job,row['original_payload_sha256'],model_sha,prompt,thresholds[prompt],recipe,local_id)
            # Refresh a whole class: its individual prompt results are no longer retained.
            force=bool(refresh and (refresh_prompts is None or any(p in refresh_prompts for p in prompts_by_class[cid])))
            cache_entries.append((cache_path, None if force else _load_prompt_cache(cache_path,shape) if cache_path.is_file() else None))
        if any(cached is None for _,cached in cache_entries):
            if model is None:
                import torch
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                report_progress('semantic',0,len(rows),detail='loading SAM3',job=job)
                if not weight.is_file(): raise FileNotFoundError(f'SAM3 checkpoint not found: {weight}')
                if expected_weight and _sha(weight) != expected_weight:
                    raise ValueError('Local SAM3 checkpoint SHA256 does not match the confirmed recipe')
                model = build_sam3_image_model(checkpoint_path=str(weight), load_from_HF=False, enable_inst_interactivity=True)
                model.eval()
                for parameter in model.parameters(): parameter.requires_grad_(False)
                processor = Sam3Processor(model, confidence_threshold=recipe['model']['instance_candidate_threshold'])
            import torch
            inference_context=torch.inference_mode()
            autocast_context=torch.autocast('cuda', dtype=torch.bfloat16)
        else:
            torch=None; inference_context=nullcontext(); autocast_context=nullcontext()
        with inference_context, autocast_context:
            if any(cached is None for _,cached in cache_entries): encoder_state=processor.set_image(image)
            for prompt,(cache_path,cached) in zip(pending_prompts,cache_entries):
                force = bool(refresh and (refresh_prompts is None or prompt in refresh_prompts))
                if cached is not None:
                    ev,strong,scores_list,masks,records=cached
                    cache_counts[prompt][0]+=1
                    prompt_records[prompt]=scores_list; evidence[prompt]=ev
                    if prompt in person_prompts: actor_strong[0]=np.maximum(actor_strong[0],strong)
                    if prompt in vehicle_prompts: actor_strong[1]=np.maximum(actor_strong[1],strong)
                    if prompt in actor_prompts:
                        visual_masks.extend(masks); visual_records.extend(records)
                        items = class_instances.setdefault(prompt_class[prompt], ([], []))
                        items[0].extend(masks); items[1].extend(records)
                    continue
                cache_counts[prompt][1]+=1
                processor.reset_all_prompts(encoder_state)
                result=processor.set_text_prompt(prompt=prompt,state=encoder_state)
                masks=result['masks'].detach().cpu().numpy().reshape(-1,*shape).astype(bool)
                probs=result['masks_logits'].float().detach().cpu().numpy().reshape(-1,*shape)
                scores=result['scores'].float().detach().cpu().numpy()
                prompt_records[prompt]=scores.tolist()
                ev=np.zeros(shape,np.float32); strong_ev=np.zeros(shape,np.float32); prompt_masks=[]; records=[]
                for ix,(binary,prob,score) in enumerate(zip(masks,probs,scores)):
                    if score < thresholds[prompt]: continue
                    weighted=np.where(binary,np.clip(prob,0,1)*float(score),0)
                    ev=np.maximum(ev,weighted)
                    if prompt in actor_prompts and score >= .7: strong_ev=np.maximum(strong_ev,weighted)
                    if prompt in actor_prompts:
                        visual_masks.append(binary); ys,xs=np.where(binary)
                        bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else [0,0,0,0]
                        if score >= thresholds[prompt]:
                            prompt_masks.append(binary)
                            records.append(dict(prompt=prompt,score=float(score),bbox_xyxy=bbox,area_pixels=int(binary.sum()),source_instance_id=len(records)))
                            visual_records.append(dict(prompt=prompt,score=float(score),bbox_xyxy=bbox,area_pixels=int(binary.sum()),source_instance_id=len(visual_records)))
                ev=np.where(ev >= recipe['base_semantics']['mask_threshold'] * thresholds[prompt],ev,0)
                evidence[prompt]=ev
                if prompt in person_prompts: actor_strong[0]=np.maximum(actor_strong[0],strong_ev)
                if prompt in vehicle_prompts: actor_strong[1]=np.maximum(actor_strong[1],strong_ev)
                if prompt in actor_prompts:
                    items = class_instances.setdefault(prompt_class[prompt], ([], []))
                    items[0].extend(prompt_masks); items[1].extend(visual_records[-len(records):] if records else [])
        for cid, path in class_paths.items():
            if cid in class_results: continue
            planes = _aggregate_class(cid, evidence, actor_strong, recipe)
            masks, records = class_instances.get(cid, ([], []))
            instance_planes = [f'instance_{i:05d}' for i in range(len(masks))]
            planes.update(zip(instance_planes, masks))
            details = dict(prompt_scores={p: prompt_records[p] for p in prompts_by_class[cid]},
                           instance_planes=instance_planes, records=records)
            save_class_cache(path, planes, details)
            class_results[cid] = planes
        # Always use the same class summaries on first inference and on cache replay.
        composed_classes = dict(class_results)
        if actor_video:
            for cid in (1, 2):
                if cid in prompts_by_class:
                    composed_classes[cid] = _aggregate_class(cid, evidence, actor_strong, recipe)
        evidence, actor_strong = _expand_classes(composed_classes, recipe, shape)
        visual_masks = video_masks; visual_records = video_records
        if not actor_video:
            for cid in prompts_by_class:
                masks, records = class_instances.get(cid, ([], []))
                visual_masks.extend(masks); visual_records.extend(records)
        labels,confidence=compose_image_semantics(evidence,actor_strong,recipe)
        Image.fromarray(labels).save(mask_path)
        np.savez_compressed(conf_path,confidence=confidence.astype(np.float16))
        _render_overlay(image,labels).save(overlay_path,quality=90)
        packed=np.packbits(np.asarray(visual_masks,bool).reshape(-1,shape[0]*shape[1]),axis=1,bitorder='little') if visual_masks else np.zeros((0,(shape[0]*shape[1]+7)//8),np.uint8)
        np.savez_compressed(output_paths[5],masks_packed=packed,mask_shape=np.asarray(shape,np.int32))
        _json(output_paths[6],dict(records=visual_records))
        # Rebuild training point labels and weak targets from untouched raw geometry.
        raw_point_labels, raw_point_conf=project_labels(arrays,labels,confidence)
        train_point_labels,train_point_conf=project_labels(arrays,training_labels(labels),confidence)
        arrays['point_raw_labels']=raw_point_labels; arrays['point_raw_confidence']=raw_point_conf
        arrays['point_labels']=train_point_labels; arrays['point_confidence']=train_point_conf
        arrays['supervision_policy']=np.array('static_drivable_occupancy_v1')
        roi=np.array([-20,-20,-3,20,20,6]); xyz=arrays['xyz']
        arrays['training_roi_mask']=np.isfinite(xyz).all(1)&((xyz>=roi[:3])&(xyz<roi[3:])).all(1)
        outside=~arrays['training_roi_mask']
        train_point_labels[outside]=UNKNOWN; train_point_conf[outside]=0
        arrays['point_labels']=train_point_labels; arrays['point_confidence']=train_point_conf
        target=weak_targets(arrays,schema='zgc_image_teacher_seven_v2')
        arrays.update(target)
        occ_bits=target['weak_occupancy_bits']; occupancy=np.full((320,320),UNKNOWN,np.uint8)
        for label,occ in enumerate((0,1,2,5,4,6,3)):
            selected=occ_bits==(1<<occ); occupancy[selected]=occ
        arrays['occupancy_labels']=occupancy
        arrays['bev_labels']=teacher_bev_from_occupancy_bits(occ_bits)
        frame_path=output_paths[4]; np.savez_compressed(frame_path,**arrays)
        frame_hash=_sha(frame_path)
        row_out={**row,'path':f'frames/{stem}.npz','local_frame_id':local_id,'original_frame_id':row['frame_id'],'sha256':frame_hash,'semantic_frame_sha256':frame_hash}
        hashes={str(p.relative_to(job)):_sha(p) for p in output_paths if p != meta_path}
        meta=dict(row=row_out,source_payload_sha256=row['original_payload_sha256'],signature=signature,actor_provider_signature=actor_signature,actor_provider_signatures=actor_source_signatures,
                  original_semantics_used=False,original_semantics_policy='raw semantic ids are untrusted; labels generated fresh from SAM3',
                  prompt_scores=prompt_records,visible_points=int(arrays['visible'].sum()),fov_points=int(arrays['in_image'].sum()),
                  output_sha256=hashes)
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        _json(meta_path,meta); rows_out.append(row_out)
        report_progress('semantic',local_id+1,len(rows),frame=row['frame_id'],job=job)
    for prompt,(hits,misses) in cache_counts.items():
        cid=prompt_class.get(prompt)
        name=next((c['name'] for c in recipe.get('raw_classes',[]) if c['id']==cid),str(cid or 'refinement'))
        print(f"semantic cache class={cid if cid is not None else 'refinement'} {name} prompt={prompt!r} threshold={thresholds[prompt]:g} hits={hits} inferences={misses}",flush=True)
    _json(root/'index.json',dict(schema='fresh_sam3_zgc_v1',rows=rows_out))
    return rows_out


def run(job: Path, refresh=False, refresh_prompts=None):
    job=Path(job).resolve(); raw=job/'raw'; idx=json.loads((raw/'index.json').read_text()); rows=idx['rows']
    recipe_path=job/'recipe.json'
    if not recipe_path.is_file(): recipe_path=RECIPE_PATH
    recipe=json.loads(recipe_path.read_text())
    return _run_model(rows,raw,job,refresh,recipe,refresh_prompts,recipe_path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--job',type=Path,required=True); p.add_argument('--refresh',action='store_true')
    a=p.parse_args(argv); run(a.job,a.refresh)


if __name__=='__main__': main()
