import unittest
from unittest.mock import patch

import numpy as np

from multi_anchor_vehicles import fit, optimize_track


def observations(n):
    return [{'frame': i, 'points': np.zeros((2, 3))} for i in range(n)]


class MultiAnchorVehicleTests(unittest.TestCase):
    def test_split_is_deterministic_and_uses_v6_validation_size(self):
        obs = observations(10)
        old = np.array([0., 0., 0., 4., 2., 1.5, 0.])
        blank = {'frame_balanced_robust_edge_cost': 1., 'median_frame_edge_m': 1.,
                 'median_projected_bbox_iou': 0.8, 'frames': 8}
        with patch('multi_anchor_vehicles.anchor_score', return_value=None), \
             patch('multi_anchor_vehicles.metrics', return_value=blank):
            a = optimize_track(7, obs, old)
            b = optimize_track(7, obs, old)
        self.assertEqual(a['train_frames'], b['train_frames'])
        self.assertEqual(a['validation_frames'], b['validation_frames'])
        self.assertEqual(len(a['validation_frames']), 2)
        self.assertEqual(len(a['train_frames']), 8)

    def test_fit_uses_fixed_size_anchor_ray_and_ninety_degree_yaw_bound(self):
        box = np.array([8., 4., 1., 4., 2., 2., 0.])
        E = np.eye(4)
        E[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        K = np.array([[600., 0, 960.], [0, 600., 540.], [0, 0, 1.]])
        points = np.c_[np.full(40, 6.), np.linspace(3.2, 4.8, 40), np.ones(40)]
        points = np.r_[points, np.c_[np.linspace(6.2, 9.8, 40), np.full(40, 3.), np.ones(40)]]
        anchor = dict(frame=4, box=box, E=E, K=K, T=np.eye(4), initial_yaw=0., points=points)
        candidate, diag = fit(anchor, [anchor], box[3:6].copy())
        np.testing.assert_array_equal(candidate[3:6], box[3:6])
        camera = E[:3, :3] @ box[:3] + E[:3, 3]
        ray_origin = np.linalg.inv(E)[:3, 3]
        ray_direction = np.linalg.inv(E)[:3, :3] @ (camera / camera[2])
        self.assertLess(np.linalg.norm(np.cross(candidate[:3] - ray_origin, ray_direction)), 1e-6)
        self.assertLessEqual(abs(candidate[6] - anchor['initial_yaw']), np.pi / 2 + 1e-6)
        self.assertLess(diag['anchor_pixel_error'], 1e-6)

    def test_fit_and_anchor_selection_receive_training_observations_only(self):
        obs = observations(10)
        old = np.array([0., 0., 0., 4., 2., 1.5, 0.])
        candidate = old.copy()
        candidate[0] = 1.
        calls = {'score': [], 'fit': [], 'metrics': []}

        def score(o):
            calls['score'].append(o['frame'])
            return {'score': 1.}

        def mocked_fit(anchor, train, size):
            calls['fit'].append((anchor['frame'], [o['frame'] for o in train]))
            return candidate, {'anchor_frame': anchor['frame'], 'converged': True}

        def measured(box, frames):
            calls['metrics'].append([o['frame'] for o in frames])
            return dict(frame_balanced_robust_edge_cost=1., median_frame_edge_m=1.,
                        median_projected_bbox_iou=.8, frames=len(frames))

        with patch('multi_anchor_vehicles.anchor_score', side_effect=score), \
             patch('multi_anchor_vehicles.fit', side_effect=mocked_fit), \
             patch('multi_anchor_vehicles.metrics', side_effect=measured):
            report = optimize_track(7, obs, old)

        train_frames = set(report['train_frames'])
        validation_frames = set(report['validation_frames'])
        self.assertEqual(set(calls['score']), train_frames)
        self.assertTrue(all(anchor_frame in train_frames and set(fit_frames) == train_frames
                            for anchor_frame, fit_frames in calls['fit']))
        self.assertTrue(train_frames.isdisjoint(validation_frames))

    def test_heldout_failure_keeps_old_final_box_and_reports_candidate(self):
        obs = observations(10)
        old = np.array([0., 0., 0., 4., 2., 1.5, 0.])
        candidate = old.copy()
        candidate[0] = 1.

        def metrics_by_length(box, frames):
            if box[0] == 1. and len(frames) == 2:
                return {'frame_balanced_robust_edge_cost': 2., 'median_frame_edge_m': 2.,
                        'median_projected_bbox_iou': 0.8, 'frames': 2}
            if box[0] == 1.:
                return {'frame_balanced_robust_edge_cost': 0.5, 'median_frame_edge_m': 0.5,
                        'median_projected_bbox_iou': 0.8, 'frames': len(frames)}
            return {'frame_balanced_robust_edge_cost': 1., 'median_frame_edge_m': 1.,
                    'median_projected_bbox_iou': 0.8, 'frames': len(frames)}

        with patch('multi_anchor_vehicles.anchor_score', return_value={'score': 1.}), \
             patch('multi_anchor_vehicles.fit', return_value=(candidate, {'anchor_frame': 0, 'anchor_pixel_error': 0.})), \
             patch('multi_anchor_vehicles.metrics', side_effect=metrics_by_length):
            report = optimize_track(7, obs, old)
        self.assertFalse(report['adopted'])
        self.assertEqual(report['status'], 'validation_failed')
        np.testing.assert_array_equal(report['final_box_map'], old)
        np.testing.assert_array_equal(report['candidate_box_map'], candidate)

    def test_passed_validation_saves_candidate_as_final_box(self):
        obs = observations(10)
        old = np.array([0., 0., 0., 4., 2., 1.5, 0.])
        candidate = old.copy()
        candidate[0] = 1.

        def measured(box, frames):
            cost = .5 if box[0] == 1. else 1.
            return dict(frame_balanced_robust_edge_cost=cost, median_frame_edge_m=cost,
                        median_projected_bbox_iou=.8, frames=len(frames))

        with patch('multi_anchor_vehicles.anchor_score', return_value={'score': 1.}), \
             patch('multi_anchor_vehicles.fit', return_value=(candidate, {'anchor_frame': 0, 'converged': True})), \
             patch('multi_anchor_vehicles.metrics', side_effect=measured):
            report = optimize_track(7, obs, old)
        self.assertTrue(report['adopted'])
        np.testing.assert_array_equal(report['final_box_map'], candidate)
        np.testing.assert_array_equal(old, [0., 0., 0., 4., 2., 1.5, 0.])


if __name__ == '__main__':
    unittest.main()
