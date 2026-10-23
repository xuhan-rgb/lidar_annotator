"""Integration coverage for prompt-cache migration to class-level replay."""
import hashlib
import json
import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from . import fresh_semantics as fs
from .recipe_config import default_recipe


class ClassCachePipelineTests(unittest.TestCase):
    shape = (12, 16)

    def _fixture(self, root, actor_video=False):
        job = Path(root) / 'job'
        raw = Path(root) / 'raw'
        job.mkdir(); raw.mkdir()
        image_path = raw / 'frame.jpg'
        Image.fromarray(np.full((*self.shape, 3), 90, np.uint8)).save(image_path)
        xyz = np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]], np.float32)
        raw_path = raw / '000000.npz'
        np.savez_compressed(raw_path, xyz=xyz,
                            pixels_uv=np.array([[2., 2.], [7., 5.], [12., 9.]], np.float32),
                            camera_depth=np.array([1., 2., 3.], np.float32),
                            in_image=np.ones(3, bool))
        source_sha = hashlib.sha256(b'one raw source payload').hexdigest()
        row = dict(frame_id='frame-17', original_payload_sha256=source_sha,
                   npz_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                   path=raw_path.name, image_path=str(image_path))
        recipe = default_recipe()
        if actor_video:
            (job / 'backend.json').write_text(json.dumps({'actor_mask_source': 'sam3_video'}))
        return job, raw, [row], recipe

    def _legacy_prompt_caches(self, job, row, recipe):
        thresholds = fs._prompt_thresholds(recipe)
        prompts = fs._semantic_prompts(recipe)
        model_sha = recipe['model'].get('sha256', recipe['model'].get('weight_sha256')) or 'missing'
        for cid, class_prompts in prompts.items():
            for prompt_index, prompt in enumerate(class_prompts):
                evidence = np.zeros(self.shape, np.float32)
                evidence[1 + prompt_index:5 + prompt_index, 2:8] = np.float32(.72)
                strong = evidence.copy() if cid in (1, 2) else np.zeros(self.shape, np.float32)
                masks, records = [], []
                if cid in (1, 2):
                    mask = np.zeros(self.shape, bool)
                    mask[2 + prompt_index:5 + prompt_index, 3:7] = True
                    masks = [mask]
                    records = [dict(prompt=prompt, score=.86,
                                    bbox_xyxy=[3, 2 + prompt_index, 7, 5 + prompt_index],
                                    area_pixels=int(mask.sum()), source_instance_id=prompt_index)]
                cache = fs._prompt_cache_path(job, row['original_payload_sha256'], model_sha,
                                              prompt, thresholds[prompt], recipe, 0)
                fs._save_prompt_cache(cache, evidence, strong, [.86], masks, records)

    def _run(self, job, raw, rows, recipe, actor_video=False):
        occupancy = np.zeros((320, 320), np.uint8)
        with mock.patch.object(fs, 'weak_targets', return_value={'weak_occupancy_bits': occupancy}), \
             mock.patch.object(fs, 'report_progress'):
            if actor_video:
                from . import actor_video as actor_module
                def read_actor(_job, prompt, _local_id, _recipe):
                    mask = np.zeros(self.shape, bool); mask[1:4, 1:5] = True
                    cid = 1 if prompt in fs._semantic_prompts(recipe)[1] else 2
                    record = dict(prompt=prompt, score=.91, video_instance_id=100 + cid,
                                  source_instance_id=0, bbox_xyxy=[1, 1, 5, 4], area_pixels=12)
                    return dict(evidence=mask.astype(np.float32) * .8,
                                strong=mask.astype(np.float32) * .8,
                                masks=mask[None], scores=[.91], records=[record])
                with mock.patch.object(actor_module, 'read_prompt', side_effect=read_actor), \
                     mock.patch.object(actor_module, 'prompt_source_signature', return_value='stable-provider'):
                    return fs._run_model(rows, raw, job, False, recipe)
            return fs._run_model(rows, raw, job, False, recipe)

    def _fake_sam(self, weight_root):
        calls = []
        builds = []

        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
            def float(self): return self

        class Parameter:
            def __init__(self): self.frozen = False
            def requires_grad_(self, value): self.frozen = not value

        class Model:
            def __init__(self): self.parameter = Parameter()
            def eval(self): return self
            def parameters(self): return [self.parameter]

        class Processor:
            def __init__(self, model, confidence_threshold): self.model = model
            def set_image(self, image): return object()
            def reset_all_prompts(self, state): pass
            def set_text_prompt(self, prompt, state):
                calls.append(prompt)
                y = sum(map(ord, prompt)) % 8
                x = sum(ord(char) * (i + 1) for i, char in enumerate(prompt)) % 12
                mask = np.zeros((1, *self.shape), bool)
                mask[0, y:y + 3, x:x + 4] = True
                logits = mask.astype(np.float32) * .8
                return {'masks': Tensor(mask), 'masks_logits': Tensor(logits),
                        'scores': Tensor(np.array([.9], np.float32))}

        # Bind the fixture's image shape to the processor without introducing a GPU dependency.
        Processor.shape = self.shape
        torch = types.ModuleType('torch')
        torch.bfloat16 = object()
        torch.inference_mode = contextlib.nullcontext
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
        sam3 = types.ModuleType('sam3'); sam3.__path__ = []
        sam3_model = types.ModuleType('sam3.model'); sam3_model.__path__ = []
        builder = types.ModuleType('sam3.model_builder')
        def build_sam3_image_model(**kwargs):
            model = Model(); builds.append(model); return model
        builder.build_sam3_image_model = build_sam3_image_model
        processor_module = types.ModuleType('sam3.model.sam3_image_processor')
        processor_module.Sam3Processor = Processor
        modules = {'torch': torch, 'sam3': sam3, 'sam3.model': sam3_model,
                   'sam3.model_builder': builder,
                   'sam3.model.sam3_image_processor': processor_module}
        return modules, calls, builds

    def test_cold_inference_replay_and_targeted_class_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            job, raw, rows, recipe = self._fixture(temp)
            weight_root = Path(temp) / 'model'
            weight_root.mkdir()
            weight = weight_root / 'sam3.pt'
            weight.write_bytes(b'fake checkpoint for offline test')
            recipe['model']['sha256'] = hashlib.sha256(weight.read_bytes()).hexdigest()
            recipe_path = Path(temp) / 'recipe.json'
            recipe_path.write_text(json.dumps(recipe))
            modules, prompt_calls, model_builds = self._fake_sam(weight_root)
            occupancy = np.zeros((320, 320), np.uint8)
            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(fs, 'MODEL_CACHE', weight_root), \
                 mock.patch.object(fs, 'weak_targets', return_value={'weak_occupancy_bits': occupancy}), \
                 mock.patch.object(fs, 'report_progress'):
                fs._run_model(rows, raw, job, False, recipe, recipe_path=recipe_path)
                self.assertEqual(len(prompt_calls), sum(map(len, fs._semantic_prompts(recipe).values())))
                self.assertEqual(len(model_builds), 1)
                self.assertTrue(all(parameter.frozen for parameter in model_builds[0].parameters()))
                self.assertTrue(list((job / 'data/class_cache').glob('*/**/*.npz')))
                self.assertFalse((job / 'data/prompt_cache').exists())
                cold_outputs = self._outputs(job)
                cache_files = {p: p.read_bytes() for p in (job / 'data/class_cache').rglob('*.npz')}

                (job / 'data/labels/zgc/metadata/000000.json').unlink()
                with mock.patch.object(fs, '_load_prompt_cache', side_effect=AssertionError('legacy cache read')):
                    fs._run_model(rows, raw, job, False, recipe, recipe_path=recipe_path)
                self.assertEqual(len(prompt_calls), sum(map(len, fs._semantic_prompts(recipe).values())))
                self.assertEqual(len(model_builds), 1)
                self._assert_same_outputs(cold_outputs, self._outputs(job))

                before_refresh_calls = len(prompt_calls)
                (job / 'data/labels/zgc/metadata/000000.json').unlink()
                fs._run_model(rows, raw, job, True, recipe, refresh_prompts=['building'],
                              recipe_path=recipe_path)
                self.assertEqual(prompt_calls[before_refresh_calls:], ['building', 'wall', 'shed'])
                self.assertEqual(len(model_builds), 2)
                self._assert_same_outputs(cold_outputs, self._outputs(job))
                changed_class_path = fs._class_cache_path(job, rows[0]['original_payload_sha256'],
                                                          recipe['model']['sha256'], 3, recipe,
                                                          fs._prompt_thresholds(recipe), 0)
                self.assertIn(changed_class_path, cache_files)
                self.assertTrue(changed_class_path.is_file())
                for path, contents in cache_files.items():
                    if path != changed_class_path:
                        self.assertEqual(contents, path.read_bytes())

    @staticmethod
    def _outputs(job):
        root = job / 'data/labels/zgc'
        paths = [root / 'image_masks/000000.png', root / 'confidence/000000.npz',
                 root / 'frames/000000.npz', job / 'data/visual_instances/000000.npz',
                 job / 'data/visual_instances/000000.json', root / 'overlay/000000.jpg']
        result = {}
        for path in paths:
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    result[str(path.relative_to(job))] = {key: archive[key].copy() for key in archive.files}
            else:
                result[str(path.relative_to(job))] = path.read_bytes()
        return result

    def _assert_same_outputs(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for path in expected:
            before, after = expected[path], actual[path]
            if isinstance(before, dict):
                self.assertEqual(before.keys(), after.keys(), path)
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key], err_msg=f'{path}:{key}')
            else:
                self.assertEqual(before, after, path)

    def test_legacy_prompt_cache_migrates_then_replays_without_legacy_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            job, raw, rows, recipe = self._fixture(temp)
            self._legacy_prompt_caches(job, rows[0], recipe)
            self._run(job, raw, rows, recipe)
            expected = self._outputs(job)
            self.assertTrue(list((job / 'data/class_cache').glob('*/**/*.npz')))

            import shutil
            shutil.rmtree(job / 'data/prompt_cache')
            (job / 'data/labels/zgc/metadata/000000.json').unlink()
            with mock.patch.object(fs, '_load_prompt_cache', side_effect=AssertionError('legacy cache read')):
                self._run(job, raw, rows, recipe)
            self._assert_same_outputs(expected, self._outputs(job))

    def test_actor_video_replay_preserves_provider_ids_and_order(self):
        with tempfile.TemporaryDirectory() as temp:
            job, raw, rows, recipe = self._fixture(temp, actor_video=True)
            self._legacy_prompt_caches(job, rows[0], recipe)
            self._run(job, raw, rows, recipe, actor_video=True)
            expected = self._outputs(job)
            records = json.loads(expected['data/visual_instances/000000.json'])['records']
            self.assertTrue(records)
            self.assertEqual([101 if r['prompt'] in fs._semantic_prompts(recipe)[1] else 102
                              for r in records], [r['video_instance_id'] for r in records])
            (job / 'data/labels/zgc/metadata/000000.json').unlink()
            self._run(job, raw, rows, recipe, actor_video=True)
            self._assert_same_outputs(expected, self._outputs(job))


if __name__ == '__main__':
    unittest.main()
