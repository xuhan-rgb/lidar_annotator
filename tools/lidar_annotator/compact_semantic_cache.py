"""Migrate completed jobs to class caches, verifying labels before optional cleanup."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from . import fresh_semantics as fs
from .actor_video import read_prompt
from .class_semantic_cache import load_class_cache, save_class_cache


def _artifacts(job, stems=None):
    roots = [job / name for name in ('data/labels', 'data/visual_instances', 'labels', 'review', 'review_local')]
    roots += [job / name for name in ('annotations.json', 'label_export.json', 'validation.json')]
    info = json.loads((job / 'job.json').read_text())
    if info.get('dataset_output'): roots.append(Path(info['dataset_output']) / 'labels')
    return {str(p): fs._sha(p) for root in roots for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file() and (stems is None or p.stem in stems)}


def _equal(expected, actual, description):
    if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
        raise ValueError(f'Exact comparison failed: {description}')


def _frame(args):
    job, recipe, local_id, row = args
    shape = np.asarray(Image.open(row['image_path'])).shape[:2]
    groups = fs._semantic_prompts(recipe); thresholds = fs._prompt_thresholds(recipe)
    model_sha = recipe['model']['sha256']
    evidence = {}; strong = np.zeros((2, *shape), np.float32)
    scores = {}; video_masks = []; video_records = []; old = []
    for cid, prompts in groups.items():
        for prompt in prompts:
            if cid in (1, 2):
                item = read_prompt(job, prompt, local_id, recipe)
                evidence[prompt] = item['evidence']; scores[prompt] = item['scores']
                strong[cid - 1] = np.maximum(strong[cid - 1], item['strong'])
                video_masks.extend(item['masks']); video_records.extend(item['records'])
            else:
                path = fs._prompt_cache_path(job, row['original_payload_sha256'], model_sha,
                                             prompt, thresholds[prompt], recipe, local_id)
                item = fs._load_prompt_cache(path, shape)
                if item is None: raise ValueError(f'Missing/corrupt prompt cache: {path}')
                evidence[prompt], _, scores[prompt], _, _ = item
                old.append(str(path))
    labels, confidence = fs.compose_image_semantics(evidence, strong, recipe)
    root = job / 'data/labels/zgc'; stem = f'{local_id:06d}'
    _equal(np.asarray(Image.open(root / 'image_masks' / f'{stem}.png')), labels, f'{stem} saved image labels')
    with np.load(root / 'confidence' / f'{stem}.npz') as z:
        _equal(z['confidence'], confidence.astype(np.float16), f'{stem} saved image confidence')
    restored = {}; class_paths = []
    for cid in groups:
        if cid in (1, 2):
            restored[cid] = fs._aggregate_class(cid, evidence, strong, recipe)
            continue
        planes = fs._aggregate_class(cid, evidence, strong, recipe)
        path = fs._class_cache_path(job, row['original_payload_sha256'], model_sha, cid, recipe, thresholds, local_id)
        details = dict(prompt_scores={p: scores[p] for p in groups[cid]}, instance_planes=[], records=[])
        save_class_cache(path, planes, details)
        loaded = load_class_cache(path, shape)
        if loaded is None: raise ValueError(f'Cannot restore class cache: {path}')
        restored[cid], metadata = loaded
        if metadata != details: raise ValueError(f'Class metadata changed: {path}')
        for name in planes: _equal(planes[name], restored[cid][name], f'{stem} class {cid}/{name}')
        class_paths.append(str(path))
    expanded, restored_strong = fs._expand_classes(restored, recipe, shape)
    new_labels, new_confidence = fs.compose_image_semantics(expanded, restored_strong, recipe)
    _equal(labels, new_labels, f'{stem} old/new labels')
    _equal(confidence, new_confidence, f'{stem} old/new float32 confidence')
    with np.load(root / 'frames' / f'{stem}.npz', allow_pickle=False) as z:
        arrays = dict(z)
    for name, image_labels in [('point_raw', labels), ('point', fs.training_labels(labels))]:
        points, point_conf = fs.project_labels(arrays, image_labels, new_confidence)
        if name == 'point':
            outside = ~arrays['training_roi_mask']; points[outside] = fs.UNKNOWN; point_conf[outside] = 0
        _equal(arrays[name + '_labels'], points, f'{stem} {name} labels')
        _equal(arrays[name + '_confidence'], point_conf, f'{stem} {name} confidence')
    targets = fs.weak_targets(arrays, schema='zgc_image_teacher_seven_v2')
    for name, value in targets.items(): _equal(arrays[name], value, f'{stem} {name}')
    _equal(arrays['bev_labels'], fs.teacher_bev_from_occupancy_bits(targets['weak_occupancy_bits']), f'{stem} BEV labels')
    with np.load(job / 'data/visual_instances' / f'{stem}.npz') as z:
        packed = np.packbits(np.asarray(video_masks, bool).reshape(-1, shape[0] * shape[1]), axis=1, bitorder='little') if video_masks else np.zeros((0, (shape[0] * shape[1] + 7) // 8), np.uint8)
        _equal(z['masks_packed'], packed, f'{stem} instance masks')
    if json.loads((job / 'data/visual_instances' / f'{stem}.json').read_text())['records'] != video_records:
        raise ValueError(f'{stem} instance IDs/records differ')
    meta = json.loads((root / 'metadata' / f'{stem}.json').read_text())
    if meta['prompt_scores'] != scores: raise ValueError(f'{stem} diagnostic scores differ')
    return dict(frame=stem, metadata_sha256=meta['metadata_sha256'], old=old, class_paths=class_paths,
                image_sha256=hashlib.sha256(labels.tobytes()).hexdigest(), confidence_sha256=hashlib.sha256(confidence.tobytes()).hexdigest(),
                weak_target_arrays=len(targets))


def run(job, workers=4, cleanup=False, limit=None):
    job = Path(job).resolve(); started = time.time()
    with (job / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads((job / 'backend.json').read_text()).get('actor_mask_source') != 'sam3_video':
            raise ValueError('Existing-job migration currently requires the confirmed SAM3 video actor workflow')
        recipe = json.loads((job / 'recipe.json').read_text())
        rows = json.loads((job / 'raw/index.json').read_text())['rows']
        if limit is not None:
            if cleanup: raise ValueError('Cannot clean up after a partial test')
            rows = rows[:limit]  # Pipeline smoke test only, never a population statistic.
        stems = {f'{i:06d}' for i in range(len(rows))} if limit is not None else None
        print('Hashing retained label and visualization artifacts…', flush=True)
        before = _artifacts(job, stems)
        print(f'Recorded {len(before)} artifact hashes; replaying frames…', flush=True)
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_frame, [(job, recipe, i, row) for i, row in enumerate(rows)], chunksize=1):
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(rows):
                    print(f'Exact verification {len(results)}/{len(rows)} frames; {time.time()-started:.1f}s', flush=True)
        print('Checking retained artifact hashes again…', flush=True)
        after = _artifacts(job, stems)
        if before != after: raise ValueError('An existing label or visualization artifact changed during migration')
        old_paths = [Path(p) for result in results for p in result['old']]
        old_files = [p for path in old_paths for p in (path, path.with_suffix('.json'))]
        old_bytes = sum(p.stat().st_blocks * 512 for p in old_files)
        new_files = [p for result in results for path in result['class_paths'] for p in (Path(path), Path(path).with_suffix('.npz.json'))]
        new_bytes = sum(p.stat().st_blocks * 512 for p in new_files)
        signature = hashlib.sha256()
        for path in (job / 'recipe.json', Path(fs.__file__), fs.HERE / 'class_semantic_cache.py'):
            signature.update(path.read_bytes())
        report = dict(status='verified', frames=len(rows), compared_existing_artifacts=len(before),
                      old_files=len(old_files), old_allocated_bytes=old_bytes, new_files=len(new_files), new_allocated_bytes=new_bytes,
                      existing_artifact_sha256=before, frame_results=results, signature=signature.hexdigest(),
                      metadata_sha256={x['frame']: x['metadata_sha256'] for x in results}, cleanup=False)
        report_path = job / 'data/class_cache' / ('migration.json' if limit is None else 'smoke_verification.json')
        fs._json(report_path, report)
        if cleanup:
            # Every old entry has already been validated against its receipt and replayed.
            for path in old_files: path.unlink()
            for path in old_paths:
                if not any(path.parent.iterdir()): path.parent.rmdir()
            report['cleanup'] = True
            report['elapsed_seconds'] = time.time() - started
            fs._json(report_path, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('existing_artifact_sha256', 'frame_results', 'metadata_sha256')}, indent=2), flush=True)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.job, args.workers, args.cleanup, args.limit)
