"""Class cache must preserve the existing semantic composer exactly."""
import copy
import json
import unittest
import numpy as np
from . import fresh_semantics as fs


class ClassSemanticTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads(fs.RECIPE_PATH.read_text())

    def check_recipe(self, recipe):
        rng = np.random.default_rng(20261019)
        shape = (19, 23)
        evidence = {p: rng.choice(np.array([0, .125, .25, .5, .7, .9], np.float32), shape)
                    for ps in fs._semantic_prompts({**recipe, 'disabled_classes': []}).values() for p in ps}
        strong = rng.choice(np.array([0, .7, .9], np.float32), (2, *shape))
        classes = {cid: fs._aggregate_class(cid, evidence, strong, recipe)
                   for cid in fs._semantic_prompts(recipe)}
        expanded, restored_strong = fs._expand_classes(classes, recipe, shape)
        expected = fs.compose_image_semantics(evidence, strong, recipe)
        # Disabled actors must also be absent in the original provider input.
        for cid in recipe.get('disabled_classes', []):
            if cid in (1, 2): strong[cid - 1] = 0
        expected = fs.compose_image_semantics(evidence, strong, recipe)
        actual = fs.compose_image_semantics(expanded, restored_strong, recipe)
        for old, new in zip(expected, actual):
            self.assertEqual(old.dtype, new.dtype)
            self.assertEqual(old.tobytes(), new.tobytes())

    def test_default_and_custom_rules_keep_exact_output(self):
        self.check_recipe(self.recipe)
        from .composition_rules import default_composition_rules
        recipe = copy.deepcopy(self.recipe)
        recipe['composition_rules'] = default_composition_rules(recipe)
        recipe['composition_rules']['priorities'].append(dict(enabled=True, source=8,
            targets=[0, 3, 255], evidence='class', condition='ge_current', compare=[], minimum_score=.3))
        self.check_recipe(recipe)

    def test_disabled_classes_and_vote_thresholds(self):
        for disabled in ([0, 2, 9], [1, 3, 8], list(range(10))):
            recipe = copy.deepcopy(self.recipe); recipe['disabled_classes'] = disabled
            self.check_recipe(recipe)
        recipe = copy.deepcopy(self.recipe)
        recipe['roof_union']['minimum_canopy_pixel_support'] = 3
        recipe['ground_refinement']['minimum_pixel_prompt_support'] = 3
        self.check_recipe(recipe)

    def test_class_key_is_local_and_invalidates_threshold_and_model(self):
        recipe = self.recipe; thresholds = fs._prompt_thresholds(recipe)
        def path(cid, config=recipe, scores=thresholds, model='model'):
            return fs._class_cache_path('/job', 'source', model, cid, config, scores, 0)
        original = path(3)
        changed = copy.deepcopy(recipe); changed['base_semantics']['prompts']['5'].append('new hedge')
        self.assertEqual(original, path(3, changed))
        changed['base_semantics']['prompts']['3'].append('new wall')
        self.assertNotEqual(original, path(3, changed))
        scores = dict(thresholds); scores['building'] = .8
        self.assertNotEqual(original, path(3, scores=scores))
        self.assertNotEqual(original, path(3, model='other'))


if __name__ == '__main__':
    unittest.main()
