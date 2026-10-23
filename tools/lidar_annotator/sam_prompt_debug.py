"""Image-only SAM3 prompt experiments, isolated from annotation outputs."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .recipe_config import load_recipe


def source_image(source, frame):
    """Read only the original JPEG from an LMDB frame; no projection."""
    import lmdb
    from .lmdb_input import frame_records
    root, rows = frame_records(source)
    row = next((r for r in rows if int(r['global_frame_id']) == frame), None)
    if row is None:
        raise ValueError(f'采集目录没有第 {frame} 帧')
    shard = root / 'shards' / f"shard_{int(row['shard_id']):06d}.lmdb"
    with lmdb.open(str(shard), readonly=True, lock=False, create=False) as env:
        with env.begin() as txn:
            payload = txn.get(row['lmdb_key'].encode())
    if payload is None:
        raise ValueError('原始帧载荷不存在')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return Image.open(io.BytesIO(data['image_jpeg_bytes'].tobytes())).convert('RGB')


def prompt_thresholds(recipe):
    from .fresh_semantics import _semantic_prompts
    prompts = _semantic_prompts(recipe)
    thresholds = {p: recipe['base_semantics']['score_threshold'] for ps in prompts.values() for p in ps}
    if 0 in prompts:
        thresholds.update(dict.fromkeys(recipe['ground_refinement']['prompts'], recipe['ground_refinement']['minimum_instance_score']))
        for name in ('manhole_covers', 'small_debris'):
            thresholds[recipe[name]['prompt']] = recipe[name]['minimum_instance_score']
    if 8 in prompts:
        thresholds[recipe['ground_objects']['prompt']] = recipe['ground_objects']['minimum_instance_score']
    if 9 in prompts:
        roof = recipe['roof_union']
        thresholds[roof['roof_prompt']] = roof['roof_threshold']
        thresholds.update(dict.fromkeys(roof['canopy_prompts'], roof['canopy_threshold']))
    return thresholds


def render_result(image, labels, recipe):
    palette = np.zeros((256, 3), np.uint8)
    for item in recipe['raw_classes']:
        palette[item['id']] = item['rgb']
    rgb = np.asarray(image).copy()
    known = labels != 255
    rgb[known] = (rgb[known] * .45 + palette[labels[known]] * .55).astype(np.uint8)
    return Image.fromarray(rgb), Image.fromarray(palette[labels])


def run(image_path, recipe_path, output, prompt=None, threshold=.25, repeat_timing=False):
    if repeat_timing and not prompt:
        raise ValueError('两次计时仅用于单条提示词测试')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from .fresh_semantics import compose_image_semantics, _sha
    from tools.autolabel.runtime_paths import MODEL_CACHE

    image_path, output = Path(image_path), Path(output)
    recipe = load_recipe(recipe_path)
    image = Image.open(image_path).convert('RGB')
    thresholds = {prompt: threshold} if prompt else prompt_thresholds(recipe)
    if not thresholds or any(not p.strip() for p in thresholds):
        raise ValueError('至少填写一条非空提示词')
    output.mkdir(parents=True, exist_ok=True)
    weight = MODEL_CACHE / 'sam3.pt'
    expected = recipe['model'].get('sha256')
    if expected and _sha(weight) != expected:
        raise ValueError('SAM3 权重与配置哈希不一致')
    torch.set_num_threads(4)
    print('正在加载 SAM3…', flush=True)
    model = build_sam3_image_model(checkpoint_path=str(weight), load_from_HF=False, enable_inst_interactivity=True).eval()
    processor = Sam3Processor(model, confidence_threshold=min(recipe['model']['instance_candidate_threshold'], min(thresholds.values())))
    evidence = {}
    strong = np.zeros((2, image.height, image.width), np.float32)
    report = dict(image=str(image_path), image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(), mode='single_prompt' if prompt else 'image_pipeline', actor_provider='single-image SAM3; no video tracking', labels_promoted=False, prompts=[])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        if repeat_timing:
            torch.cuda.synchronize()
            started = time.perf_counter()
            first_state = processor.set_image(image)
            torch.cuda.synchronize()
            first_feature_ms = (time.perf_counter() - started) * 1000
            processor.reset_all_prompts(first_state)
            torch.cuda.synchronize()
            started = time.perf_counter()
            processor.set_text_prompt(prompt=prompt, state=first_state)
            torch.cuda.synchronize()
            first_decode_ms = (time.perf_counter() - started) * 1000
            report['timing_runs'] = [dict(run=1, image_feature_ms=first_feature_ms, decode_ms=first_decode_ms)]
            print(f'第 1 次：图像特征提取 {first_feature_ms:.2f} ms；decode {first_decode_ms:.2f} ms', flush=True)
            del first_state
        prefix = '第 2 次：' if repeat_timing else ''
        torch.cuda.synchronize()
        started = time.perf_counter()
        state = processor.set_image(image)
        torch.cuda.synchronize()
        report['image_feature_ms'] = (time.perf_counter() - started) * 1000
        print(f'{prefix}图像特征提取：{report["image_feature_ms"]:.2f} ms（含图像预处理，不含模型加载）', flush=True)
        for ordinal, (text, minimum) in enumerate(thresholds.items()):
            processor.reset_all_prompts(state)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = processor.set_text_prompt(prompt=text, state=state)
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - started) * 1000
            print(f'{prefix}提示词 {text!r} decode：{decode_ms:.2f} ms（含文本编码与 mask 后处理，不含文件保存）', flush=True)
            if repeat_timing:
                report['timing_runs'].append(dict(run=2, image_feature_ms=report['image_feature_ms'], decode_ms=decode_ms))
            masks = result['masks'].cpu().numpy().reshape(-1, image.height, image.width).astype(bool)
            probabilities = result['masks_logits'].float().cpu().numpy().reshape(masks.shape)
            scores = result['scores'].float().cpu().numpy()
            ev = np.zeros((image.height, image.width), np.float32)
            actor = ev.copy()
            union = np.zeros(ev.shape, bool)
            for mask, probability, score in zip(masks, probabilities, scores):
                if score < minimum:
                    continue
                value = np.where(mask, np.clip(probability, 0, 1) * float(score), 0)
                ev = np.maximum(ev, value)
                union |= mask
                if score >= .7:
                    actor = np.maximum(actor, value)
            evidence[text] = np.where(ev >= recipe['base_semantics']['mask_threshold'] * minimum, ev, 0)
            for i, cid in enumerate((1, 2)):
                if text in recipe['base_semantics']['prompts'][str(cid)]:
                    strong[i] = np.maximum(strong[i], actor)
            name = f'prompt_{ordinal:03d}'
            Image.fromarray(union.astype(np.uint8) * 255).save(output / f'{name}_mask.png')
            rgb = np.asarray(image).copy()
            rgb[union] = (rgb[union] * .55 + np.array([255, 0, 200]) * .45).astype(np.uint8)
            Image.fromarray(rgb).save(output / f'{name}_overlay.png')
            report['prompts'].append(dict(prompt=text, threshold=minimum, scores=scores.tolist(), accepted=int((scores >= minimum).sum()), mask_pixels=int(union.sum()), prefix=name, decode_ms=decode_ms))
            print(f'{ordinal+1}/{len(thresholds)}：{text}；检出 {report["prompts"][-1]["accepted"]} 个区域', flush=True)
    if not prompt:
        labels, confidence = compose_image_semantics(evidence, strong, recipe)
        overlay, colors = render_result(image, labels, recipe)
        overlay.save(output / 'final_overlay.png'); colors.save(output / 'final_colors.png')
        Image.fromarray(labels).save(output / 'final_labels.png')
        np.savez_compressed(output / 'confidence.npz', confidence=confidence.astype(np.float16))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('图像提示词调试完成', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--threshold', type=float, default=.25)
    parser.add_argument('--repeat-timing', action='store_true')
    args = parser.parse_args()
    run(args.image, args.recipe, args.output, args.prompt, args.threshold, args.repeat_timing)


if __name__ == '__main__':
    main()
