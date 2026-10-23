import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from common import SOURCE, atomic_json, read_source_image, render_overlay, sha
from union_policy import PROMPTS, THRESHOLDS, apply_union

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,MODEL_CACHE
ROOT = DATA
WEIGHT = MODEL_CACHE/'sam3.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rows = json.loads((SOURCE / 'zgc/index.json').read_text())['rows']
    if args.smoke:
        rows = [r for r in rows if Path(r['path']).stem in ('000000', '000012', '000017', '000021', '000221', '000298', '000411', '000440')]
        assert len(rows) == 8
    out = ROOT / ('smoke_labels' if args.smoke else 'labels')
    dest = out / 'zgc'
    for folder in ('image_masks', 'confidence', 'overlay', 'metadata'):
        (dest / folder).mkdir(parents=True, exist_ok=True)
    policy = {'base_labels': str(SOURCE), 'base_index_sha256': sha(SOURCE / 'zgc/index.json'),
              'prompts': PROMPTS, 'instance_thresholds': THRESHOLDS, 'mask_threshold': .5,
              'rule': 'building intersect (roof OR at least two canopy prompts); preserve previous accepted roof masks',
              'minimum_canopy_prompt_support': 2,
              'image_to_training_labels': [0, 1, 2, 3, 4, 5, 6, 6, 6, 4],
              'roof_occupancy_label': 4, 'preserve_ground_and_nonbuilding_classes': True,
              'visualization_xy_range_m': [-20, -20, 20, 20], 'person_rgb': [0, 255, 255],
              'manual_ground_truth': False, 'weight': str(WEIGHT), 'weight_sha256': sha(WEIGHT)}
    policy_path = out / 'policy.json'
    if policy_path.exists():
        assert json.loads(policy_path.read_text()) == policy
    atomic_json(policy_path, policy)
    started = time.time()

    def status(state, completed, **extra):
        item = dict(state=state, pid=os.getpid(), completed=completed, total=len(rows),
                    elapsed_seconds=round(time.time() - started, 1), **extra)
        atomic_json(out / 'status.json', item)
        print(json.dumps(item), flush=True)

    status('loading_sam3', 0)
    torch.set_num_threads(4)
    model = build_sam3_image_model(checkpoint_path=str(WEIGHT), load_from_HF=False,
                                   enable_inst_interactivity=True)
    processor = Sam3Processor(model, confidence_threshold=.25)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        text_features = {(p,): model.backbone.forward_text([p], device='cuda') for p in PROMPTS}
        model.backbone.forward_text = lambda texts, device=None, **kwargs: text_features[tuple(texts)]
        for ordinal, row in enumerate(rows, 1):
            stem = Path(row['path']).stem
            meta_path = dest / 'metadata' / (stem + '.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['source'] == row
                for key in ('mask', 'confidence', 'overlay'):
                    assert sha(dest / meta[key]) == meta[key + '_sha256']
                status('generating', ordinal, frame=stem, resumed=True)
                continue
            parent = json.loads((SOURCE / 'zgc/metadata' / (stem + '.json')).read_text())
            for key in ('mask', 'confidence'):
                assert sha(SOURCE / 'zgc' / parent[key]) == parent[key + '_sha256']
            assert sha(SOURCE / 'zgc' / row['path']) == row['sha256']
            old = np.array(Image.open(SOURCE / 'zgc' / parent['mask']))
            with np.load(SOURCE / 'zgc' / parent['confidence']) as a:
                old_confidence = a['confidence'].astype(np.float32)
            image = read_source_image(row)
            assert old.shape == (image.height, image.width)
            state = processor.set_image(image)
            evidence, counts = {}, {}
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(prompt=prompt, state=state)
                masks = result['masks'].cpu().numpy().reshape(-1, *old.shape)
                probabilities = result['masks_logits'].float().cpu().numpy().reshape(-1, *old.shape)
                scores = result['scores'].float().cpu().numpy()
                counts[prompt] = scores.tolist()
                value = np.zeros(old.shape, np.float32)
                for mask, probability, score in zip(masks, probabilities, scores):
                    if score >= THRESHOLDS[prompt]:
                        value = np.maximum(value, np.where(mask, probability * score, 0))
                evidence[prompt] = value
            labels, confidence, selected = apply_union(old, old_confidence, evidence)
            protected = old != 3
            np.testing.assert_array_equal(labels[protected], old[protected])
            np.testing.assert_array_equal(confidence[protected], old_confidence[protected])
            assert np.isfinite(confidence).all() and (confidence >= 0).all() and (confidence <= 1).all()
            assert (confidence[labels == 255] == 0).all()
            mask_path = dest / 'image_masks' / (stem + '.png')
            confidence_path = dest / 'confidence' / (stem + '.npz')
            overlay_path = dest / 'overlay' / (stem + '.jpg')
            Image.fromarray(labels).save(mask_path)
            np.savez_compressed(confidence_path, confidence=confidence.astype(np.float16))
            render_overlay(image, labels, 'zgc', stem, .55).save(overlay_path, quality=70)
            meta = {'source': row, 'base_metadata': str(SOURCE / 'zgc/metadata' / (stem + '.json')),
                    'old_mask': str(SOURCE / 'zgc' / parent['mask']), 'old_mask_sha256': parent['mask_sha256'],
                    'prompts': counts,
                    'added_roof_pixels': int(selected.sum()),
                    'roof_pixels': int((labels == 9).sum()),
                    'class_pixel_counts': np.bincount(labels[labels < 10], minlength=10).tolist(),
                    'overlay_mask_alpha': .55, 'overlay_layout': 'left original, right mask overlay',
                    'overlay_jpeg_quality': 70}
            for key, path in [('mask', mask_path), ('confidence', confidence_path), ('overlay', overlay_path)]:
                meta[key] = str(path.relative_to(dest))
                meta[key + '_sha256'] = sha(path)
            atomic_json(meta_path, meta)
            status('generating', ordinal, frame=stem, roof_pixels=meta['roof_pixels'])
            del state, result
    status('images_complete', len(rows))


if __name__ == '__main__':
    main()
