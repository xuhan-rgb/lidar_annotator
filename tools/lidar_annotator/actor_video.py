"""Cache native SAM3 video masks and identities for actor prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'tools/autolabel/semantic_zgc'))
from common import read_source_image  # noqa: E402

ALGORITHM_VERSION = 'sam3_native_video_actor_v3'
MAX_GAP_NS = 2_000_000_000


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2))
    tmp.replace(path)


def split_temporal_chunks(timestamps_ns):
    ts = np.asarray(timestamps_ns, np.int64)
    if not len(ts): return []
    chunks = []; start = 0
    for i in range(1, len(ts)):
        delta = int(ts[i]) - int(ts[i - 1])
        if delta <= 0 or delta > MAX_GAP_NS:
            chunks.append(list(range(start, i))); start = i
    chunks.append(list(range(start, len(ts))))
    return chunks


def _checkpoint():
    sys.path.insert(0, str(HERE.parent))
    from runtime_paths import MODEL_CACHE
    return Path(MODEL_CACHE) / 'sam3.pt'


def _load_model():
    import torch
    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(checkpoint_path=str(_checkpoint()), load_from_HF=False,
                                   strict_state_dict_loading=True, compile=False).cuda().eval()
    original = model.tracker.init_state
    def init_state(*args, **kwargs):
        kwargs['offload_state_to_cpu'] = True
        return original(*args, **kwargs)
    model.tracker.init_state = init_state
    return model


def _cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _frame_signature(raw_path, row, timestamp):
    h = hashlib.sha256()
    h.update(_sha(raw_path).encode())
    h.update(str(row.get('frame_id', '')).encode())
    h.update(str(timestamp).encode())
    return h.hexdigest()


def _chunk_key(job, rows, timestamps, indices, prompt, recipe, model_sha):
    raw = Path(job) / 'raw'
    identity = dict(algorithm=ALGORITHM_VERSION, prompt=prompt, model_sha256=model_sha,
                    score_threshold=float(recipe['base_semantics'].get('score_threshold', .25)),
                    mask_threshold=float(recipe['base_semantics'].get('mask_threshold', .5)),
                    sources=[_frame_signature(raw / rows[i]['path'], rows[i], timestamps[i]) for i in indices])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _actor_root(job, prompt):
    ph = hashlib.sha256(prompt.encode()).hexdigest()
    return Path(job) / 'data/actor_video' / ph, ph


def _valid_chunk(path, key, indices):
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('chunk_key') != key or manifest.get('local_indices') != indices: return False
        return all(_sha(path / name) == digest for name, digest in manifest['files'].items())
    except (OSError, ValueError, KeyError):
        return False


def _write_chunk(path, key, indices, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=path.name + '.tmp-', dir=path.parent))
    try:
        files = {}
        for index, (masks, scores, ids, shape, source_sig) in zip(indices, frames):
            name = f'frames/{index:06d}.npz'; target = temp / name; target.parent.mkdir(parents=True, exist_ok=True)
            h, w = shape
            packed = np.packbits(np.asarray(masks, bool).reshape(len(ids), h*w), axis=1, bitorder='little')
            np.savez_compressed(target, masks_packed=packed, mask_shape=np.asarray(shape, np.int32),
                                scores=np.asarray(scores, np.float32), track_ids=np.asarray(ids, np.int64),
                                source_signature=np.asarray(source_sig))
            files[name] = _sha(target)
            metadata_name = f'frames/{index:06d}.json'
            metadata_path = temp / metadata_name
            _json(metadata_path, dict(local_index=index, source_signature=source_sig,
                                      mask_shape=list(shape), mask_count=len(ids)))
            files[metadata_name] = _sha(metadata_path)
        _json(temp / 'manifest.json', dict(chunk_key=key, local_indices=indices, files=files,
              score_semantics='native SAM3 object confidence attached to binary video mask; not pixel posterior'))
        if path.exists(): shutil.rmtree(path)
        temp.replace(path)
    finally:
        if temp.exists(): shutil.rmtree(temp)


def _infer_chunk(model, job, rows, indices, prompt, offset, signatures):
    import torch
    raw = Path(job) / 'raw'; image_dir = Path(tempfile.mkdtemp(prefix='actor-video-'))
    try:
        shapes = {}
        for fi, index in enumerate(indices):
            image_row = dict(rows[index])
            if image_row.get('image_path'):
                image_path = Path(image_row['image_path'])
                if not image_path.is_absolute() and not image_path.exists(): image_path = raw / image_path
                image_row['image_path'] = str(image_path)
            else:
                image_row.pop('image_path', None)
            image = read_source_image(image_row)
            shapes[fi] = (image.height, image.width)
            image.save(image_dir / f'{fi:05d}.jpg', quality=95)
        state = model.init_state(str(image_dir), offload_video_to_cpu=True)
        model.add_prompt(state, frame_idx=0, text_str=prompt)
        frames = [None] * len(indices)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for fi, outputs in model.propagate_in_video(state, start_frame_idx=0,
                    max_frame_num_to_track=len(indices)-1, reverse=False):
                fi = int(fi); ids = _cpu(outputs['out_obj_ids']).astype(np.int64).reshape(-1)
                masks = _cpu(outputs['out_binary_masks']).astype(bool)
                if masks.ndim == 4: masks = masks[:, 0]
                if not len(ids): masks = np.empty((0, *shapes[fi]), bool)
                if 'out_probs' not in outputs:
                    raise KeyError('SAM3 video output is missing native object confidence out_probs')
                scores = _cpu(outputs['out_probs']).astype(np.float32).reshape(-1)
                if len(scores) != len(ids): raise ValueError('SAM3 native object confidence count does not match object IDs')
                track_ids = ids + np.int64(offset)
                frames[fi] = (masks, scores, track_ids, masks.shape[-2:], signatures[indices[fi]])
                if isinstance(state, dict) and 'cached_frame_outputs' in state:
                    for old in list(state['cached_frame_outputs']):
                        if old < fi - 32: del state['cached_frame_outputs'][old]
        for fi, frame in enumerate(frames):
            if frame is None: frames[fi] = (np.empty((0, *shapes[fi]), bool), np.empty(0, np.float32), np.empty(0, np.int64), shapes[fi], signatures[indices[fi]])
        return frames
    finally:
        shutil.rmtree(image_dir, ignore_errors=True)


def _prompts(recipe):
    disabled = set(recipe.get('disabled_classes', [])); base = recipe['base_semantics']['prompts']
    return list(base.get('1', [])) * (1 if 1 not in disabled else 0) + list(base.get('2', [])) * (1 if 2 not in disabled else 0)


def run(job, recipe, refresh=False, refresh_prompts=None):
    """Populate reusable video masks for active person and vehicle prompts."""
    job = Path(job).resolve(); recipe_path = job / 'recipe.json'
    if recipe is None:
        recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else json.loads((REPO / 'tools/autolabel/zgc_mask_recipe_v1/recipe.json').read_text())
    raw = job / 'raw'; index = json.loads((raw / 'index.json').read_text()); rows = index['rows']
    timestamps = []
    for row in rows:
        with np.load(raw / row['path'], allow_pickle=False) as data: timestamps.append(int(np.asarray(data['lidar_timestamp_ns']).item()))
    chunks = split_temporal_chunks(timestamps); prompts = _prompts(recipe)
    refresh_set = set(refresh_prompts if refresh_prompts is not None else prompts) if refresh else set()
    results = {}; model = None; model_sha = _sha(_checkpoint())
    signatures = [_frame_signature(raw / row['path'], row, timestamps[i]) for i, row in enumerate(rows)]
    for prompt in prompts:
        root, prompt_hash = _actor_root(job, prompt); records = []
        # An interrupted update must never leave an older source signature looking current.
        (root / 'manifest.json').unlink(missing_ok=True)
        for chunk_index, indices in enumerate(chunks):
            key = _chunk_key(job, rows, timestamps, indices, prompt, recipe, model_sha)
            chunk_path = root / 'chunks' / f'{chunk_index:06d}-{key[:16]}'
            if prompt not in refresh_set and _valid_chunk(chunk_path, key, indices):
                records.append(dict(chunk_key=key, path=str(chunk_path.relative_to(root)), reused=True)); continue
            if model is None: model = _load_model()
            offset = (int(prompt_hash[:8], 16) & 0x7fffffff) * (1 << 32) + (chunk_index * (1 << 20))
            frames = _infer_chunk(model, job, rows, indices, prompt, offset, signatures)
            _write_chunk(chunk_path, key, indices, frames)
            records.append(dict(chunk_key=key, path=str(chunk_path.relative_to(root)), reused=False))
        source_signature = hashlib.sha256(json.dumps(dict(prompt_hash=prompt_hash,
            chunks=[dict(key=x['chunk_key'],files=json.loads((root/x['path']/'manifest.json').read_text())['files']) for x in records]), sort_keys=True).encode()).hexdigest()
        manifest = dict(prompt=prompt, prompt_hash=prompt_hash, algorithm=ALGORITHM_VERSION,
                        model_sha256=model_sha, chunks=records, frame_count=len(rows),
                        source_signature=source_signature,
                        score_semantics='native SAM3 object confidence attached to binary video mask; not pixel posterior')
        _json(root / 'manifest.json', manifest)
        results[prompt] = dict(status='complete', chunks=len(chunks), reused_chunks=sum(x['reused'] for x in records),
                               source_signature=source_signature)
    return dict(status='complete', frame_count=len(rows), prompts=results, disabled_classes=recipe.get('disabled_classes', []),
                identity_namespace='prompt hash plus temporal chunk index plus native SAM3 track ID')


def prompt_source_signature(job, prompt, recipe=None):
    """Return a cheap aggregate identity for semantic-frame cache signatures."""
    root, _ = _actor_root(job, prompt)
    try:
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('prompt') != prompt: return None
        if recipe is not None and prompt not in _prompts(recipe): return None
        for chunk in manifest['chunks']:
            path = root / chunk['path']
            if not _valid_chunk(path, chunk['chunk_key'], json.loads((path / 'manifest.json').read_text())['local_indices']): return None
        return manifest['source_signature']
    except (OSError, ValueError, KeyError):
        return None


def read_prompt(job, prompt, row_local_index, recipe=None):
    """Load cached binary masks and derive semantic evidence for one raw row."""
    root, _ = _actor_root(job, prompt); local = int(row_local_index)
    manifest = json.loads((root / 'manifest.json').read_text())
    if recipe is not None and prompt not in _prompts(recipe): raise KeyError(f'inactive actor prompt: {prompt}')
    for chunk in manifest['chunks']:
        cm = json.loads((root / chunk['path'] / 'manifest.json').read_text())
        if local in cm['local_indices']:
            file = root / chunk['path'] / 'frames' / f'{local:06d}.npz'
            with np.load(file, allow_pickle=False) as z:
                shape = tuple(map(int, z['mask_shape'])); packed=z['masks_packed']; scores=z['scores']; ids=z['track_ids']
                masks=np.unpackbits(packed,axis=1,count=shape[0]*shape[1],bitorder='little').reshape(-1,*shape).astype(bool)
                source_sig=str(z['source_signature'].item())
            threshold=float((recipe or {}).get('base_semantics',{}).get('score_threshold',.25))
            selected=scores >= threshold; strong_selected=scores >= .7
            evidence=np.zeros(shape,np.float32); strong=np.zeros(shape,np.float32)
            for mask, score in zip(masks[selected],scores[selected]): evidence[mask]=np.maximum(evidence[mask],score)
            for mask, score in zip(masks[strong_selected],scores[strong_selected]): strong[mask]=np.maximum(strong[mask],score)
            kept=masks[selected]; kept_scores=scores[selected]; kept_ids=ids[selected]
            records=[]
            for mask,score,track_id in zip(kept,kept_scores,kept_ids):
                y,x=np.where(mask); bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else [0,0,0,0]
                records.append(dict(prompt=prompt,score=float(score),bbox_xyxy=bbox,area_pixels=int(mask.sum()),source_instance_id=int(track_id),track_id=int(track_id)))
            return dict(evidence=evidence,strong=strong,masks=kept,records=records,scores=kept_scores.tolist(),source_signature=source_sig)
    raise IndexError(local)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--job',type=Path,required=True); parser.add_argument('--refresh',action='store_true'); args=parser.parse_args()
    recipe_path=args.job/'recipe.json'; recipe=json.loads(recipe_path.read_text()) if recipe_path.exists() else None
    print(json.dumps(run(args.job,recipe,args.refresh),indent=2))


if __name__ == '__main__': main()
