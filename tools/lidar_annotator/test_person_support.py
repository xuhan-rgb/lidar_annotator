import unittest

import numpy as np

from tools.lidar_annotator.person_support import select_person_support


class PersonSupportTests(unittest.TestCase):
    def setUp(self):
        core = np.array([[0, 0, z] for z in range(7)], dtype=float)
        self.xyz = np.vstack([core, [[2, 0, 1], [2.1, 0, 2]], [[0.1, 0.1, 50]]])
        self.support = np.arange(len(self.xyz))
        self.labels = np.ones(len(self.xyz), dtype=int)
        self.center = [0, 0, 0]

    def test_robust_core_rejects_xy_outliers_without_z_clipping(self):
        got = select_person_support(self.xyz, self.support, self.labels, self.center)
        np.testing.assert_array_equal(got['kept_indices'], [0, 1, 2, 3, 4, 5, 6, 9])
        np.testing.assert_array_equal(got['rejected_indices'], [7, 8])
        self.assertFalse(got['review_required'])

    def test_saved_person_support_is_preserved_even_outside_new_core(self):
        got = select_person_support(self.xyz, self.support, self.labels, self.center, saved_ids=[7])
        self.assertIn(7, got['kept_indices'])
        self.assertNotIn(7, got['rejected_indices'])
        self.assertTrue(got['saved_support_used'])

    def test_fail_open_cases(self):
        for center in (None, [10, 0, 0]):
            got = select_person_support(self.xyz, self.support, self.labels, center)
            np.testing.assert_array_equal(got['kept_indices'], self.support)
            self.assertEqual(len(got['rejected_indices']), 0)
            self.assertTrue(got['review_required'])
        got = select_person_support(self.xyz[:4], np.arange(4), self.labels[:4], self.center)
        np.testing.assert_array_equal(got['kept_indices'], np.arange(4))
        self.assertTrue(got['review_required'])

    def test_only_original_person_support_can_be_changed(self):
        labels = self.labels.copy()
        labels[7] = 2
        got = select_person_support(self.xyz, self.support, labels, self.center)
        self.assertNotIn(7, got['kept_indices'])
        self.assertNotIn(7, got['rejected_indices'])
        self.assertIn(0, got['kept_indices'])

    def test_saved_indices_invalid_fail_open_but_outside_support_ignored(self):
        for saved in ([-1], [len(self.xyz)], [1.5]):
            got = select_person_support(self.xyz, self.support, self.labels, self.center, saved)
            self.assertTrue(got['review_required'])
            self.assertEqual(len(got['rejected_indices']), 0)
        got = select_person_support(self.xyz, [0, 1, 2, 3, 4, 5, 6, 7, 8], self.labels,
                                    self.center, [9])
        self.assertFalse(got['review_required'])
        self.assertFalse(got['saved_support_used'])

    def test_inputs_are_unchanged(self):
        xyz, support, labels = self.xyz.copy(), self.support.copy(), self.labels.copy()
        select_person_support(self.xyz, self.support, self.labels, self.center, [7])
        np.testing.assert_array_equal(self.xyz, xyz)
        np.testing.assert_array_equal(self.support, support)
        np.testing.assert_array_equal(self.labels, labels)

    def test_incoherent_cloud_does_not_preserve_distant_background(self):
        xyz=np.array([[0,0,1],[.1,.1,5],[3,0,1],[6,0,1],[9,0,1],[12,0,1],[15,0,1],[18,0,1]],float)
        labels=np.array([1,1,1,1,1,1,1,3])
        result=select_person_support(xyz,np.arange(len(xyz)),labels,[0,0,1],saved_ids=[2])
        np.testing.assert_array_equal(result['kept_indices'],[0,1,2])
        np.testing.assert_array_equal(result['rejected_indices'],[3,4,5,6])
        self.assertTrue(result['review_required'])
        self.assertEqual(result['fallback_max_distance_m'],1.25)


if __name__ == '__main__':
    unittest.main()
