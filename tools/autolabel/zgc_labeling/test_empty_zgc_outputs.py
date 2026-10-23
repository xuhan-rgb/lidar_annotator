"""Synthetic empty-population regressions for map and person-track review."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_person_consistent_tracks import main as audit_person_tracks
from visualize_vehicle_map import main as visualize_vehicle_map


class EmptyZgcOutputTests(unittest.TestCase):
    def make_map_fixture(self, root, observation=None):
        root = Path(root)
        source = root / 'source'
        (source / 'frames').mkdir(parents=True)
        labels = root / 'labels'
        (labels / 'metadata').mkdir(parents=True)
        xyz = np.array([[1., 1., 0.]]) if observation else np.empty((0, 3), np.float32)
        semantics = np.array([1], np.uint8) if observation else np.empty((0,), np.uint8)
        np.savez(source / 'frames/000000.npz', xyz=xyz, point_labels=semantics)
        (labels / 'metadata/000000.json').write_text(json.dumps({'observations': [observation] if observation else []}))
        index = source / 'index.json'
        index.write_text(json.dumps({'rows': [{'path': 'frames/000000.npz'}]}))
        poses = root / 'poses.npz'
        np.savez(poses, transforms=np.eye(4, dtype=np.float32)[None])
        config = root / 'map_config.json'
        config.write_text(json.dumps({'temporal_output': str(labels), 'source_index': str(index), 'pose_npz': str(poses), 'coordinate_assumption': 'synthetic'}))
        return config, labels

    def test_vehicle_map_writes_shaped_empty_arrays_without_background_or_vehicles(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, labels = self.make_map_fixture(tmp)
            visualize_vehicle_map(['--config', str(config)])
            with np.load(labels / 'map_review/map_points.npz') as data:
                self.assertEqual(data['background_xyz_map'].shape, (0, 3))
                self.assertEqual(data['vehicle_xyz_map'].shape, (0, 3))
                self.assertEqual(data['vehicle_track_ids'].shape, (0,))
            manifest = json.loads((labels / 'map_review/manifest.json').read_text())
            self.assertIsNone(manifest['focus_track'])
            self.assertFalse(list((labels / 'map_review').glob('T*_temporal_views.png')))

    def test_vehicle_map_falls_back_when_default_focus_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, labels = self.make_map_fixture(tmp, {'kind': 'vehicle', 'track_id': 2, 'seed_id': 1, 'support_ids': [0], 'status': 'accepted_single_frame'})
            visualize_vehicle_map(['--config', str(config)])
            manifest = json.loads((labels / 'map_review/manifest.json').read_text())
            self.assertEqual(manifest['focus_track'], 2)
            self.assertTrue((labels / 'map_review/T002_temporal_views.png').exists())

    def test_person_audit_keeps_ignored_count_for_empty_retained_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temporal = root / 'labels'
            out = temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks'
            video = temporal / 'map_review/sam3_full_video'
            out.mkdir(parents=True)
            centers = out / 'centers_only.npz'
            np.savez(centers, position_available=np.array([False, False]), corrected_xy_map=np.zeros((2, 2)), sam_video_ids=np.array([4, 5]), frame_ids=np.array([0, 0]), timestamp_ns=np.array([0, 1]), instance_ids=np.array([10, 11]), original_point_lidar=np.zeros((2, 3)), corrected_point_lidar=np.zeros((2, 3)), reference_xy_map=np.zeros((2, 2)))
            (video / 'visual_presence_index.json').write_text('{}')
            (out / 'report.json').write_text(json.dumps({'reference_models': []}))
            (out / 'index.html').write_text('<html></html>')
            source = root / 'source'
            (source / 'frames').mkdir(parents=True)
            index = source / 'index.json'
            index.write_text(json.dumps({'rows': []}))
            poses = root / 'poses.npz'
            np.savez(poses, transforms=np.eye(4, dtype=np.float32)[None])
            config = root / 'audit_config.json'
            config.write_text(json.dumps({'temporal_output': str(temporal), 'source_index': str(index), 'pose_npz': str(poses)}))

            audit_person_tracks(['--config', str(config)])

            result = json.loads((out / 'integrity_audit.json').read_text())
            self.assertEqual(result['input'], 2)
            self.assertEqual(result['retained'], 0)
            self.assertEqual(result['ignored'], 2)
            self.assertIsNone(result['max_pixel_change_px'])
            self.assertIsNone(result['min_camera_depth_m'])
            self.assertIsNone(result['max_world_transform_error_m'])
            self.assertIsNone(result['max_reference_distance_m'])


if __name__ == '__main__':
    unittest.main()
