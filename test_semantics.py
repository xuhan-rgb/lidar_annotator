import json
import unittest
from pathlib import Path

import numpy as np

from tools.lidar_annotator.fresh_semantics import (
    apply_roof_union,
    confidence_for_labels,
    project_labels,
    teacher_bev_from_occupancy_bits,
    training_labels,
    zbuffer_visible,
)


ROOT = Path(__file__).resolve().parent
RECIPE = json.loads((ROOT / 'tools/autolabel/zgc_mask_recipe_v1/recipe.json').read_text())


class FreshSemanticPolicyTests(unittest.TestCase):
    def test_roof_union_requires_building_and_two_canopy_votes(self):
        labels = np.array([[3, 3, 2, 3]], np.uint8)
        conf = np.full((1, 4), .2, np.float32)
        evidence = {'roof': np.array([[0, 0, 1, 0]], np.float32),
                    'overhanging roof': np.array([[1, 1, 1, 1]], np.float32),
                    'metal canopy': np.array([[1, 0, 1, 1]], np.float32),
                    'overhead metal canopy': np.zeros((1, 4), np.float32)}
        out, updated = apply_roof_union(labels, conf, evidence, RECIPE)
        np.testing.assert_array_equal(out, [[9, 3, 2, 9]])
        self.assertAlmostEqual(float(updated[0, 0]), 1)
        self.assertAlmostEqual(float(updated[0, 1]), .2)

    def test_zbuffer_visibility_is_separate_from_fov(self):
        uv = np.array([[1.2, 1.2], [1.8, 1.9], [0, 0], [9, 9]], np.float32)
        depth = np.array([3, 4, 2, 1], np.float32)
        fov = np.array([True, True, True, False])
        visible = zbuffer_visible(uv, depth, fov, (3, 3))
        np.testing.assert_array_equal(visible, [True, False, True, False])
        labels = np.arange(9, dtype=np.uint8).reshape(3, 3)
        point, _ = project_labels(dict(pixels_uv=uv, visible=visible), labels, np.ones((3, 3), np.float32))
        np.testing.assert_array_equal(point, [4, 255, 0, 255])

    def test_training_mapping_preserves_raw_labels(self):
        raw = np.array([0, 6, 7, 8, 9, 255], np.uint8)
        train = training_labels(raw)
        np.testing.assert_array_equal(raw, [0, 6, 7, 8, 9, 255])
        np.testing.assert_array_equal(train, [0, 6, 6, 6, 4, 255])

    def test_confidence_uses_the_final_selected_class(self):
        scores = np.zeros((8, 1, 2), np.float32)
        scores[4, 0, 0] = .8; scores[5, 0, 0] = .6; scores[3, 0, 1] = .7
        labels = np.array([[5, 255]], np.uint8)
        np.testing.assert_allclose(confidence_for_labels(scores, labels), [[.6, 0]])

    def test_bev_teacher_labels_restore_xy_storage_orientation(self):
        bits = np.zeros((3, 4), np.uint8)
        bits[1, 2] = 1 << 6  # occupancy fence -> teacher class 5
        labels = teacher_bev_from_occupancy_bits(bits)
        self.assertEqual(labels.shape, (4, 3))
        self.assertEqual(labels[2, 1], 5)
        self.assertEqual(labels[1, 2], 255)


if __name__ == '__main__':
    unittest.main()
