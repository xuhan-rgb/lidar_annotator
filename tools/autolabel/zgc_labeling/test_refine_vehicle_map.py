"""Regression coverage for writing accepted refined vehicle map boxes."""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from refine_vehicle_map import run


class RefineVehicleMapTests(unittest.TestCase):
    def make_fixture(self, root, track_ids=(5, 8)):
        root = Path(root)
        source = root / 'source'
        (source / 'frames').mkdir(parents=True)
        frame_npz = source / 'frames/000000.npz'
        xyz = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        intrinsic = np.eye(3)
        extrinsic = np.eye(4)
        np.savez(frame_npz, xyz=xyz, cam_intrinsic=intrinsic, cam_extrinsic=extrinsic)
        source_index = source / 'index.json'
        source_index.write_text(json.dumps({'rows': [{'path': 'frames/000000.npz'}]}))

        temporal = root / 'temporal'
        metadata = temporal / 'metadata'
        metadata.mkdir(parents=True)
        observations = []
        for track_id in track_ids:
            observations.append({
                'kind': 'vehicle', 'status': 'accepted_temporal', 'track_id': track_id,
                'support_ids': [0, 1], 'box': [float(track_id), 0., 0., 4., 2., 1.5, 0.],
            })
        (metadata / '000000.json').write_text(json.dumps({'observations': observations}))

        history = temporal / 'map_review/history_fit'
        dedup = history / 'bev_dedup'
        dedup.mkdir(parents=True)
        old_boxes = np.array([[float(t), 0., 0., 4., 2., 1.5, 0.] for t in track_ids]).reshape(-1, 7)
        candidate_boxes = old_boxes.copy()
        candidate_boxes[:, 1] = 99.
        (history / 'report.json').write_text(json.dumps({'results': [
            {'track_id': int(t), 'candidate_box_map': b.tolist()}
            for t, b in zip(track_ids, candidate_boxes)
        ]}))
        authoritative = dedup / 'vehicle_map_boxes.npz'
        np.savez(authoritative, box_3d_map=old_boxes, track_ids=np.asarray(track_ids, dtype=np.int64))

        pose_npz = root / 'poses.npz'
        poses = np.eye(4, dtype=np.float32)[None]
        np.savez(pose_npz, transforms=poses)
        config = root / 'config.json'
        config.write_text(json.dumps({
            'source_index': str(source_index), 'temporal_output': str(temporal),
            'pose_npz': str(pose_npz),
        }))
        return config, temporal, authoritative, old_boxes, candidate_boxes, xyz, intrinsic, extrinsic, poses

    def test_writes_only_final_boxes_preserving_ids_sizes_sources_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            (config, temporal, authoritative, old_boxes, candidates, xyz, intrinsic,
             extrinsic, poses) = self.make_fixture(tmp)
            accepted = old_boxes[0].copy()
            accepted[0] += 10.
            rejected_candidate = old_boxes[1].copy()
            rejected_candidate[1] += 20.
            responses = {
                5: {'track_id': 5, 'status': 'accepted', 'adopted': True,
                    'candidate_box_map': accepted.tolist(),
                    'final_box_map': accepted.tolist()},
                8: {'track_id': 8, 'status': 'validation_failed',
                    'candidate_box_map': rejected_candidate.tolist(),
                    'adopted': False, 'final_box_map': old_boxes[1].tolist()},
            }
            source_npz = Path(json.loads(config.read_text())['source_index']).parent / 'frames/000000.npz'
            source_before = source_npz.read_bytes()
            pose_path = Path(json.loads(config.read_text())['pose_npz'])
            pose_before = pose_path.read_bytes()
            metadata_path = temporal / 'metadata/000000.json'
            metadata_before = metadata_path.read_bytes()

            with patch('refine_vehicle_map.optimize_track', side_effect=lambda tid, obs, old: responses[tid]):
                report = run(config)

            with np.load(authoritative) as data:
                np.testing.assert_array_equal(data['track_ids'], [5, 8])
                result_boxes = data['box_3d_map'].copy()
            np.testing.assert_array_equal(result_boxes[0], accepted)
            np.testing.assert_array_equal(result_boxes[1], old_boxes[1])
            self.assertFalse(np.array_equal(result_boxes[1], rejected_candidate))
            np.testing.assert_array_equal(result_boxes[:, 3:6], old_boxes[:, 3:6])
            backup_root = temporal / 'map_review/history_fit/multi_anchor/backups'
            backups = list(backup_root.glob('*/vehicle_map_boxes.npz'))
            self.assertEqual(len(backups), 1)
            with np.load(backups[0]) as data:
                np.testing.assert_array_equal(data['track_ids'], [5, 8])
                np.testing.assert_array_equal(data['box_3d_map'], old_boxes)
            self.assertTrue(report)
            audit = json.loads((temporal / 'map_review/history_fit/multi_anchor/report.json').read_text())
            self.assertEqual(audit['retained_track_ids'], [5, 8])
            self.assertEqual(audit['candidate_tracks_updated'], 1)

            self.assertEqual(source_npz.read_bytes(), source_before)
            self.assertEqual(pose_path.read_bytes(), pose_before)
            self.assertEqual(metadata_path.read_bytes(), metadata_before)
            with np.load(source_npz) as data:
                np.testing.assert_array_equal(data['xyz'], xyz)
                np.testing.assert_array_equal(data['cam_intrinsic'], intrinsic)
                np.testing.assert_array_equal(data['cam_extrinsic'], extrinsic)
            with np.load(pose_path) as data:
                np.testing.assert_array_equal(data['transforms'], poses)

            before_mtime = authoritative.stat().st_mtime_ns
            before_bytes = authoritative.read_bytes()
            time.sleep(.01)
            with patch('refine_vehicle_map.optimize_track', side_effect=lambda tid, obs, old: responses[tid]):
                run(config)
            self.assertEqual(authoritative.stat().st_mtime_ns, before_mtime)
            self.assertEqual(authoritative.read_bytes(), before_bytes)
            self.assertEqual(len(list(backup_root.glob('*/vehicle_map_boxes.npz'))), 1)

    def test_empty_history_candidates_succeeds_without_changing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, temporal, authoritative, old_boxes, _, *_ = self.make_fixture(tmp)
            history_report = temporal / 'map_review/history_fit/report.json'
            history_report.write_text(json.dumps({'results': []}))
            original = authoritative.read_bytes()
            result = run(config)
            self.assertTrue(result)
            self.assertEqual(authoritative.read_bytes(), original)

    def test_empty_retained_ids_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, temporal, authoritative, *_ = self.make_fixture(tmp, track_ids=())
            original = authoritative.read_bytes()
            result = run(config)
            self.assertTrue(result)
            self.assertEqual(authoritative.read_bytes(), original)

    def test_review_initializer_loads_authoritative_deduplicated_map_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, temporal, authoritative, old_boxes, candidates, *_ = self.make_fixture(tmp)
            np.testing.assert_array_equal(candidates[:, 1], [99., 99.])
            self.assertFalse(np.array_equal(candidates, old_boxes))
            person = temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks'
            person.mkdir(parents=True)
            np.savez(person / 'centers_only.npz', frame_ids=np.empty(0, dtype=np.int64))
            source_stub = types.ModuleType('common')
            source_stub.read_source_image = lambda row: None
            with patch.dict(sys.modules, {'common': source_stub}):
                import render_consistent_combined
                render_consistent_combined.initialize(config)

            self.assertEqual([int(track_id) for track_id, _ in render_consistent_combined.CARS], [5, 8])
            for (_, actual), expected in zip(render_consistent_combined.CARS, old_boxes):
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
