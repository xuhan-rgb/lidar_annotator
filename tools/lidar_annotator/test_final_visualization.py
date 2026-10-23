"""Regression tests for using final optimized vehicle boxes in frame and map views."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np

from .live_preview import load_frame
from .optimized_map import build_optimized_map
from .test_optimized_map import fixture as map_fixture


def add_legacy_vehicle(root):
    """Add one accepted legacy vehicle annotation whose box differs from final."""
    visual = root / 'data/visual_instances'
    visual.mkdir(parents=True, exist_ok=True)
    path = visual / '000000.json'
    path.write_text(json.dumps({'records': [dict(prompt='car', track_id=303, bbox_xyxy=[1, 1, 3, 3])]}))
    vehicle = root / 'data/vehicle/refined'
    vehicle.mkdir(parents=True)
    (vehicle / '000000.json').write_text(json.dumps({'all_detections': [
        dict(prediction_index=4, status='review_label', sam_bbox_xyxy=[1, 1, 3, 3],
             class_verification={'yolo_score': .8, 'yolo_bbox_xyxy': [.5, .5, 3.5, 3.5]})]}))
    np.savez(vehicle / '000000.npz', seed_ids=[4], box_3d=[[20, 20, 5, 2, 2, 2, 0]])


class FinalVisualizationTests(unittest.TestCase):
    @staticmethod
    def _append_points(root, points, labels):
        source = root / 'data/labels/zgc/frames/000000.npz'
        with np.load(source) as z:
            arrays = {key: z[key] for key in z.files}
        points = np.asarray(points, np.float32)
        arrays['xyz'] = np.concatenate((arrays['xyz'], points))
        arrays['point_raw_labels'] = np.concatenate((arrays['point_raw_labels'], np.asarray(labels, np.uint8)))
        arrays['pixels_uv'] = np.concatenate((arrays['pixels_uv'], np.tile([[1., 1.]], (len(points), 1))))
        arrays['camera_depth'] = np.concatenate((arrays['camera_depth'], points[:, 2]))
        for key in ('visible', 'in_image'):
            if key in arrays:
                arrays[key] = np.concatenate((arrays[key], np.ones(len(points), bool)))
        np.savez_compressed(source, **arrays)

    def test_combined_uses_final_map_boxes_and_relabels_points_inside_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = map_fixture(root)
            add_legacy_vehicle(root)
            np.savez_compressed(root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz',
                                box_3d_map=[[0, 0, 5, 2, 1, 2, np.pi / 4]], track_ids=[42])
            self._append_points(root, [[.6, 0, 5], [.9, .9, 5], [0, 0, 3.8], [0, 0, 5]], [255, 2, 255, 1])
            source = root / 'data/labels/zgc/frames/000000.npz'
            with np.load(source) as z:
                original_xyz = z['xyz'].copy()
            source_hash = __import__('hashlib').sha256(source.read_bytes()).hexdigest()
            combined = load_frame(context, 63, mode='combined')
            self.assertEqual(combined['box_ids'], [42])
            np.testing.assert_allclose(combined['boxes'][0].mean(axis=0), [0, 0], atol=1e-6)
            # The accepted old box is remote; its box cannot override the final track box.
            self.assertNotIn(4, combined['box_ids'])
            map_data = build_optimized_map(context, 0)
            palette = np.full((256, 3), 100, np.uint8)
            for item in map_data['palette']:
                palette[item['id']] = item['rgb']
            # A point in the rotated 3D box becomes vehicle; outside and below it do not.
            np.testing.assert_array_equal(combined['bev_colors'][-4:], palette[[2, 255, 255, 1]])
            np.testing.assert_array_equal(combined['camera_colors'][-4:], palette[[2, 255, 255, 1]])
            with np.load(map_data['archive']) as archive:
                first_frame = archive['source_frame_ids'] == 63
                first_labels = archive['point_raw_labels'][first_frame]
                np.testing.assert_array_equal(first_labels[-4:], [2, 255, 255, 1])
            np.testing.assert_array_equal(combined['bev_colors'], palette[first_labels])
            camera_indices = combined['camera_point_indices']
            bev_positions = np.searchsorted(combined['bev_point_indices'], camera_indices)
            np.testing.assert_array_equal(combined['camera_colors'], combined['bev_colors'][bev_positions])
            self.assertEqual(source_hash, __import__('hashlib').sha256(source.read_bytes()).hexdigest())
            with np.load(source) as z:
                np.testing.assert_array_equal(z['xyz'], original_xyz)

    def test_final_boxes_and_point_classes_match_map_including_pose_inverse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = map_fixture(root)
            add_legacy_vehicle(root)
            final_path = root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
            np.savez_compressed(final_path, box_3d_map=[[10, 3, 5, 2, 1, 2, .5]], track_ids=[42])
            map_data = build_optimized_map(context, 0)
            data = load_frame(context, 64, mode='combined')
            self.assertEqual(data['box_ids'], [42])
            self.assertEqual(set(map_data['car_ids']), set(data['box_ids']))
            inverse = np.linalg.inv(np.load(root / 'temporal/poses.npz')['transforms'][1])
            expected_center = inverse[:3, :3] @ map_data['boxes'][0, :3] + inverse[:3, 3]
            np.testing.assert_allclose(data['boxes'][0].mean(axis=0), expected_center[:2], atol=1e-6)

    def test_legacy_fallback_without_final_result_and_empty_final_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = map_fixture(root)
            add_legacy_vehicle(root)
            final_path = root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
            final_path.unlink()
            fallback = load_frame(context, 63, mode='combined')
            self.assertEqual(fallback['box_ids'], [4])
            np.savez_compressed(final_path, box_3d_map=np.array([]), track_ids=np.array([], int))
            empty = load_frame(context, 63, mode='combined')
            self.assertEqual(empty['box_ids'], [])
            # Existing vehicle-class points lose that class when no final box supports them.
            self.assertEqual(empty['bev_colors'][3].tolist(), [100, 100, 100])
            map_data = build_optimized_map(context, 0)
            with np.load(map_data['archive']) as archive:
                labels = archive['point_raw_labels'][archive['source_frame_ids'] == 63]
                self.assertEqual(int(labels[3]), 255)


if __name__ == '__main__':
    unittest.main()
