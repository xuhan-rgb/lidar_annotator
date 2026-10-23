"""Composition rule defaults and configurable class overlap semantics."""
import copy
import unittest
import numpy as np
from .recipe_config import default_recipe, validate_recipe
from .fresh_semantics import (_semantic_prompts, UNKNOWN, confidence_for_labels,
    apply_ground_refinements, apply_roof_union, compose_image_semantics)


def legacy_compose(evidence, actor_strong, recipe):
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
    # Class-specific tie/override rules confirmed in the base semantic stage.
    trees=np.isin(labels,(4,5,6)); fence_score, trunk_score=scores[5],scores[6]
    labels[trees & (fence_score>0) & (fence_score>=trunk_score)]=5
    labels[trees & (trunk_score>0) & (trunk_score>fence_score)]=6
    actor_support=np.maximum(actor_strong[0],actor_strong[1])
    unsupported=np.isin(labels,(1,2)) & (actor_support==0); labels[unsupported]=UNKNOWN
    strong_actor=actor_support>0
    labels[strong_actor]=np.where(actor_strong[0, strong_actor] >= actor_strong[1, strong_actor], 1, 2)
    confidence=confidence_for_labels(scores,labels)
    confidence[strong_actor]=np.maximum(actor_strong[0, strong_actor],actor_strong[1, strong_actor])
    # Grass wins ties against foliage/fence and preserves actors, building and trunk.
    grass=np.maximum.reduce([evidence[p] for p in recipe['base_semantics']['grass_prompts']]) if 0 in prompts_by_class else np.zeros(shape,np.float32)
    grass_sel=np.isin(labels,(0,4,5,UNKNOWN)) & (grass>0) & (grass>=np.maximum(scores[4],scores[5]))
    labels[grass_sel]=0; confidence[grass_sel]=np.maximum(confidence[grass_sel],grass[grass_sel])
    pole=np.maximum.reduce([evidence[p] for p in recipe['base_semantics']['prompts']['7']]) if 7 in prompts_by_class else np.zeros(shape,np.float32); pole_sel=np.isin(labels,(4,5,6)) & (pole>0) & (pole>=np.take_along_axis(scores,np.minimum(labels,7)[None],axis=0)[0])
    labels[pole_sel]=7; confidence[pole_sel]=pole[pole_sel]
    # Fresh predictions compete by score; old building/fence conflict handling
    # depended on previous teacher labels, which are intentionally untrusted here.
    if 0 in prompts_by_class or 8 in prompts_by_class:
        labels,confidence=apply_ground_refinements(labels,confidence,evidence,recipe,
            include_ground_objects=8 in prompts_by_class,include_refinements=0 in prompts_by_class)
    if 9 in prompts_by_class: labels,confidence=apply_roof_union(labels,confidence,evidence,recipe)
    # Ignore unconfirmed actor pixels, while retaining SAM detections for instance review.
    actors=np.maximum.reduce([evidence[p] for p in actor_prompts]) if actor_prompts else np.zeros(shape,np.float32)
    actor_area=np.isin(labels,(1,2)) & (actors==0); labels[actor_area]=UNKNOWN; confidence[actor_area]=0
    confidence[labels==UNKNOWN]=0
    return labels,confidence


from .composition_rules import composition_rules, default_composition_rules


class CompositionRuleTests(unittest.TestCase):
    def fixture(self, recipe, shape=(4, 5)):
        evidence = {p: np.zeros(shape, np.float32) for ps in _semantic_prompts(recipe).values() for p in ps}
        return evidence, np.zeros((2, *shape), np.float32)

    def test_defaults_match_legacy_on_complete_synthetic_arrays(self):
        rng = np.random.default_rng(817)
        for disabled in ([], [0], [1, 2], [8, 9], [4, 5, 6, 7], list(range(10))):
            for seed in range(4):
                recipe = default_recipe(); recipe['disabled_classes'] = disabled
                evidence, strong = self.fixture(recipe)
                for value in evidence.values():
                    value[:] = rng.choice([0., .15, .4, .7, .9], size=value.shape)
                for i, cid in enumerate((1, 2)):
                    if cid not in disabled:
                        strong[i] = rng.choice([0., .7, .9], size=strong[i].shape)
                expected = legacy_compose(evidence, strong, recipe)
                for explicit in (False, True):
                    if explicit: recipe['composition_rules'] = default_composition_rules(recipe)
                    actual = compose_image_semantics(evidence, strong, recipe)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])

    def test_subtraction_transfers_overlap_or_ignores_and_respects_disabled(self):
        recipe = default_recipe(); evidence, strong = self.fixture(recipe)
        evidence['building'][:] = .9; evidence['tree canopy'][0, 0] = .4
        recipe['composition_rules'] = dict(priorities=[], subtractions=[dict(enabled=True, source=3, subtract=[4], result='target')])
        labels, conf = compose_image_semantics(evidence, strong, recipe)
        self.assertEqual(labels[0, 0], 4); self.assertAlmostEqual(float(conf[0, 0]), .4)
        self.assertEqual(labels[0, 1], 3)
        recipe['composition_rules']['subtractions'][0]['result'] = 'ignore'
        labels, conf = compose_image_semantics(evidence, strong, recipe)
        self.assertEqual(labels[0, 0], 255); self.assertEqual(conf[0, 0], 0)
        recipe['disabled_classes'] = [4]
        labels, _ = compose_image_semantics(evidence, strong, recipe)
        self.assertTrue((labels == 3).all())

    def test_disable_default_roof_subtraction_keeps_building(self):
        recipe = default_recipe(); evidence, strong = self.fixture(recipe)
        evidence['building'][:] = .8; evidence['roof'][:] = .7
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 9).all())
        recipe['composition_rules'] = composition_rules(recipe)
        recipe['composition_rules']['subtractions'][0]['enabled'] = False
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 3).all())

    def test_priority_threshold_comparison_and_execution_order(self):
        recipe = default_recipe(); evidence, strong = self.fixture(recipe)
        evidence['building'][:] = .9; evidence['metal fence'][:] = .4; evidence['tree trunk'][:] = .3
        rule = dict(enabled=True, source=5, targets=[3, 6], evidence='class', condition='always', compare=[], minimum_score=.5)
        recipe['composition_rules'] = dict(priorities=[rule], subtractions=[])
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 3).all())
        rule['minimum_score'] = .2
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 5).all())
        rule['condition'] = 'ge_current'
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 3).all())
        rule['condition'] = 'always'
        second = dict(enabled=True, source=6, targets=[3, 5], evidence='class', condition='always', compare=[], minimum_score=0)
        recipe['composition_rules']['priorities'] = [rule, second]
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 6).all())
        recipe['composition_rules']['priorities'] = [second, rule]
        self.assertTrue((compose_image_semantics(evidence, strong, recipe)[0] == 5).all())

    def test_validation_rejects_invalid_or_ambiguous_rules(self):
        recipe = default_recipe(); recipe['composition_rules'] = composition_rules(recipe)
        for change in (
            lambda r: r['subtractions'][0].update(subtract=[3]),
            lambda r: r['priorities'][0].update(targets=[]),
            lambda r: r['priorities'][0].update(condition='invalid'),
            lambda r: r['priorities'][0].update(minimum_score=float('nan')),
            lambda r: r['priorities'][0].update(evidence='strong'),
        ):
            modified = copy.deepcopy(recipe); change(modified['composition_rules'])
            with self.assertRaises(ValueError):validate_recipe(modified)


if __name__ == '__main__':
    unittest.main()
