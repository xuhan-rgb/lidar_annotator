import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from . import fresh_semantics as fs


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            'model': {'instance_candidate_threshold': .25, 'inference_autocast': 'cuda bfloat16'},
            'base_semantics': {'mask_threshold': .5},
        }
        self.shape = (5, 7)

    def test_only_changed_prompt_misses_and_targeted_refresh_runs_once(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = {'road': 0, 'person': 0}
            def infer(prompt):
                def run():
                    calls[prompt] += 1
                    ev = np.zeros(self.shape, np.float32); ev[1:3, 2:4] = .5
                    return ev, np.zeros_like(ev), [.8], [], []
                return run
            paths = {p: fs._prompt_cache_path(temp, 'source', 'model', p, .25, self.recipe, 0)
                     for p in calls}
            for prompt in calls:
                fs.cached_prompt_result(paths[prompt], self.shape, False, infer(prompt))
            # Editing one prompt yields a distinct cache key; the other prompt remains reusable.
            changed = fs._prompt_cache_path(temp, 'source', 'model', 'road edited', .25, self.recipe, 0)
            fs.cached_prompt_result(paths['person'], self.shape, False, infer('person'))
            fs.cached_prompt_result(changed, self.shape, False, infer('road'))
            self.assertEqual(calls, {'road': 2, 'person': 1})
            fs.cached_prompt_result(paths['person'], self.shape, True, infer('person'))
            self.assertEqual(calls['person'], 2)

    def test_corrupt_artifact_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fs._prompt_cache_path(temp, 'source', 'model', 'person', .25, self.recipe, 0)
            calls = []
            def infer():
                calls.append(1)
                ev = np.zeros(self.shape, np.float32); ev[2, 3] = .75
                return ev, ev.copy(), [.8], [], []
            fs.cached_prompt_result(path, self.shape, False, infer)
            path.write_bytes(b'corrupt')
            fs.cached_prompt_result(path, self.shape, False, infer)
            self.assertEqual(len(calls), 2)

    def test_disabled_classes_remove_base_and_auxiliary_prompts(self):
        recipe = json.loads(fs.RECIPE_PATH.read_text())
        recipe['disabled_classes'] = [0, 2, 8, 9]
        active = fs._semantic_prompts(recipe)
        self.assertNotIn(0, active)
        self.assertNotIn(2, active)
        self.assertNotIn(9, active)
        self.assertNotIn(recipe['base_semantics']['grass_prompts'][0], sum(active.values(), []))
        self.assertNotIn(recipe['ground_refinement']['prompts'][0], sum(active.values(), []))
        self.assertNotIn(recipe['ground_objects']['prompt'], sum(active.values(), []))
        self.assertNotIn(recipe['roof_union']['roof_prompt'], sum(active.values(), []))

    def test_actor_provider_signature_tracks_each_prompt_source(self):
        source_signatures={'person':'person-r1','pedestrian':'person-r1'}
        first=fs._actor_manifest_signature(source_signatures,['person','pedestrian'])
        source_signatures['person']='person-r2'
        self.assertNotEqual(first,fs._actor_manifest_signature(source_signatures,['person','pedestrian']))


if __name__ == '__main__':
    unittest.main()
