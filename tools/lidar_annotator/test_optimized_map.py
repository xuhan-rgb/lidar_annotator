"""All-frame alignment, optimized annotations, full-point retention and map controls."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QPushButton
from .optimized_map import build_optimized_map, transform_points, voxel_indices, export_map_html, box_corners, person_trajectory_segments, person_track_starts
from .live_preview import load_frame
from .map_view import MapCanvas
from .qt_app import AnnotationWindow
from .cli import STAGES


def fixture(root):
    raw = root / 'raw'; labels = root / 'data/labels/zgc'; video = root / 'temporal/map_review/sam3_full_video'
    for path in (raw / 'frames', labels / 'frames', labels / 'image_masks', video / 'masks', video / 'interior_half_review/consistent_tracks', root / 'temporal/map_review/history_fit/bev_dedup'):
        path.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), (100, 100, 100)).save(raw / 'image.png')
    rows = []
    xyz = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5], [0, -1, 5], [21, 0, 5]], np.float32)
    uv = np.array([[4, 4], [6, 4], [4, 6], [4, 2], [46, 4]], np.float32)
    for local, frame in enumerate((63, 64)):
        arrays = dict(xyz=xyz, pixels_uv=uv, camera_depth=np.full(5, 5.), visible=np.array([1, 1, 1, 1, 0], bool),
                      in_image=np.array([1, 1, 1, 1, 0], bool), point_raw_labels=np.array([1, 1, 3, 2, 255], np.uint8),
                      cam_intrinsic=np.array([[10, 0, 4], [0, 10, 4], [0, 0, 1]]), cam_extrinsic=np.eye(4))
        np.savez_compressed(labels / 'frames' / f'{local:06d}.npz', **arrays)
        np.savez_compressed(raw / 'frames' / f'{local:06d}.npz', **arrays)
        rows.append(dict(path=f'frames/{local:06d}.npz', original_frame_id=frame, local_frame_id=local, image_path=str(raw / 'image.png')))
        semantic = np.full((8, 8), 255, np.uint8); semantic[4, 4] = 1; semantic[4, 6] = 1; semantic[6, 4] = 3; semantic[2, 4] = 2
        Image.fromarray(semantic).save(labels / 'image_masks' / f'{local:06d}.png')
        mask = np.zeros((1, 8, 8), bool); mask[0, 4, [4, 6]] = True
        np.savez_compressed(video / 'masks' / f'{local:06d}.npz', mask_shape=[8, 8], object_ids=[7], masks_packed=np.packbits(mask.reshape(1, -1), axis=1, bitorder='little'))
    for path in (raw / 'index.json', labels / 'index.json'):path.write_text(json.dumps(dict(rows=rows)))
    poses = np.tile(np.eye(4), (2, 1, 1)); poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; poses[1, :3, 3] = [10, 2, 0]
    np.savez_compressed(root / 'temporal/poses.npz', transforms=poses, segments=[0, 0])
    np.savez_compressed(video / 'interior_half_review/consistent_tracks/centers_only.npz', frame_ids=[0, 1], sam_video_ids=[7, 7],
                        position_available=[True, True], original_point_lidar=[[.5, 0, 5], [.5, 0, 5]], corrected_point_lidar=[[0, 0, 5], [0, 0, 5]])
    np.savez_compressed(root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz', box_3d_map=[[10, 3, 1, 2, 1, 2, .5]], track_ids=[42])
    return dict(output=str(root)), xyz, poses


class OptimizedMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app = QApplication.instance() or QApplication([])

    def test_all_frames_transform_points_and_only_optimized_targets_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context, xyz, poses = fixture(root)
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.npz')}
            progress = []
            data = build_optimized_map(context, 0, lambda n, total: progress.append((n, total)))
            self.assertEqual(data['frame_count'], 2); self.assertEqual(data['point_count'], 8)
            np.testing.assert_allclose(data['xyz'], np.concatenate([xyz[:4], transform_points(xyz[:4], poses[1])]))
            np.testing.assert_allclose(data['person_xyz'], [[0, 0, 5], [10, 2, 5]])
            np.testing.assert_array_equal(data['person_frames'], [63, 64]); np.testing.assert_array_equal(data['car_ids'], [42])
            np.testing.assert_array_equal(data['trajectory_frames'], [63, 64])
            np.testing.assert_allclose(data['trajectory'], poses[:, :3, 3])
            # Two person points do not provide enough evidence for semantic deletion.
            self.assertEqual(data['ignored_count'], 0)
            # Old vehicle-colored points above/outside the final 3D box are ignored.
            np.testing.assert_array_equal(data['labels'], [1, 1, 3, 255, 1, 1, 3, 255])
            self.assertEqual(progress, [(1, 2), (2, 2)])
            with np.load(data['archive']) as archive:
                self.assertEqual(len(archive['xyz_map']), 8)
                np.testing.assert_array_equal(archive['source_frame_ids'], [63] * 4 + [64] * 4)
            for path, digest in originals.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_voxels_are_display_only_and_prefer_actor_labels(self):
        xyz = np.array([[0, 0, 0], [.01, .01, .01], [1, 1, 1]])
        ids = voxel_indices(xyz, np.array([255, 1, 3]), .15)
        self.assertEqual(set(ids), {1, 2})
        with tempfile.TemporaryDirectory() as directory:
            context, _, _ = fixture(Path(directory))
            data = build_optimized_map(context, 100.)
            self.assertLess(len(data['xyz']), data['point_count'])
            with np.load(data['archive']) as archive:self.assertEqual(len(archive['xyz_map']), 8)

    def test_no_vehicle_detections_and_empty_targets_still_generate_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context, _, _ = fixture(root)
            path = root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
            # Existing zero-detection files store [] as shape (0,), not (0, 7).
            np.savez_compressed(path, box_3d_map=np.array([]), track_ids=np.array([], int))
            data = build_optimized_map(context, 0)
            self.assertEqual(data['boxes'].shape, (0, 7))
            self.assertEqual(data['point_count'], 8)
            self.assertTrue(export_map_html(data).is_file())
            people = root / 'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
            np.savez_compressed(people, frame_ids=np.array([], int), sam_video_ids=np.array([], int),
                position_available=np.array([], bool), original_point_lidar=np.empty((0, 3)), corrected_point_lidar=np.empty((0, 3)))
            empty = build_optimized_map(context, 0)
            self.assertEqual(empty['person_xyz'].shape, (0, 3)); self.assertEqual(empty['point_count'], 8)
            self.assertTrue(export_map_html(empty).is_file())

    def test_missing_optimization_partial_frames_and_cancellation_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context, _, _ = fixture(root)
            with self.assertRaises(InterruptedError):build_optimized_map(context, cancelled=lambda: True)
            index = root / 'data/labels/zgc/index.json'
            value = json.loads(index.read_text()); value['rows'][0]['local_frame_id'] = 100; index.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, '匹配'):build_optimized_map(context)
            (root / 'temporal/poses.npz').unlink()
            with self.assertRaisesRegex(ValueError, '时序优化'):build_optimized_map(context)

    def test_map_uses_all_current_task_frames_not_other_recording_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context, _, _ = fixture(root)
            (root / 'job.json').write_text(json.dumps(dict(source='/test/recording', frame_ids=[63, 64])))
            with patch('tools.lidar_annotator.lmdb_input.frame_records') as recording:
                data = build_optimized_map(context)
            recording.assert_not_called(); self.assertEqual(data['frame_count'], 2)
            self.assertEqual(set(data['person_frames']), {63, 64})
            for index in (root / 'raw/index.json', root / 'data/labels/zgc/index.json'):
                content = json.loads(index.read_text()); content['rows'][1]['original_frame_id'] = 70
                index.write_text(json.dumps(content))
            sparse = build_optimized_map(context)
            np.testing.assert_array_equal(sparse['segments'], [0, 1])
            self.assertEqual(set(sparse['person_frames']), {63, 70})

    def test_map_matches_each_combined_view_point_colors_positions_and_targets(self):
        for raw_labels in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); context, _, poses = fixture(root)
                # Combined views require accepted detections for their 2D/person overlays.
                visual = root / 'data/visual_instances'; visual.mkdir(parents=True)
                people = root / 'data/person/metadata'; people.mkdir(parents=True)
                for local in range(2):
                    (visual / f'{local:06d}.json').write_text(json.dumps(dict(records=[dict(
                        prompt='person', track_id=7, bbox_xyxy=[3, 3, 7, 6])])))
                    (people / f'{local:06d}.json').write_text(json.dumps(dict(instances=[dict(
                        sam_source_instance_id=7, status='accepted_xy_from_all_exclusive_mask_points',
                        person_class_confirmed=True, yolo_match=dict(score=.9, bbox_xyxy=[3, 3, 7, 6]))])))
                if not raw_labels:
                    for path in (root / 'data/labels/zgc/frames').glob('*.npz'):
                        with np.load(path) as z:arrays = {k: z[k] for k in z.files if k != 'point_raw_labels'}
                        np.savez_compressed(path, **arrays)
                data = build_optimized_map(context, 0)
                palette = np.full((256, 3), 100, np.uint8)
                for item in data['palette']:palette[item['id']] = item['rgb']
                for local, frame in enumerate((63, 64)):
                    combined = load_frame(context, frame, mode='combined')
                    chosen = slice(local * 4, (local + 1) * 4)
                    np.testing.assert_array_equal(palette[data['labels'][chosen]], combined['bev_colors'])
                    inverse = np.linalg.inv(poses[local])
                    np.testing.assert_allclose(transform_points(data['xyz'][chosen], inverse)[:, :2], combined['bev_points'], atol=1e-6)
                    np.testing.assert_allclose(transform_points(data['person_xyz'][[local]], inverse)[:, :2], combined['centres'], atol=1e-6)
                    self.assertEqual(set(data['car_ids']), set(combined['box_ids']))

    def test_person_tracks_sort_frames_and_do_not_bridge_gaps_or_pose_segments(self):
        data = dict(person_ids=np.array([1, 2, 1, 1, 1, 1]), person_frames=np.array([2, 0, 0, 1, 4, 5]),
                    person_segments=np.array([0, 0, 0, 0, 0, 1]))
        paths = list(person_trajectory_segments(data))
        self.assertEqual(len(paths), 1); self.assertEqual(paths[0][0], 1)
        np.testing.assert_array_equal(paths[0][1], [2, 3, 0])
        self.assertEqual([(int(tid), int(index)) for tid, index in person_track_starts(data)], [(1, 2), (2, 1)])

    def test_map_projection_and_html_include_people_vehicles_and_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = build_optimized_map(fixture(Path(directory))[0])
            canvas = MapCanvas(); canvas.resize(800, 600); canvas.set_map(data)
            pixels = canvas.project(np.array([[0, 0], [1, 0], [0, 1]]))
            self.assertLess(pixels[1, 1], pixels[0, 1]); self.assertLess(pixels[2, 0], pixels[0, 0])
            self.assertEqual(box_corners(data['boxes'][0]).shape, (8, 3))
            html = export_map_html(data).read_text()
            self.assertIn('V42', html); self.assertIn('Map X (m)', html); self.assertIn('plotly', html)
            with patch('plotly.graph_objects.Figure.write_html', autospec=True) as writer:
                export_map_html(data)
            figure = writer.call_args.args[0]
            start = next(trace for trace in figure.data if trace.name == '起始帧')
            self.assertEqual(start.mode, 'markers'); self.assertEqual(len(start.x), 1)
            self.assertIsNone(start.text)
            frames = next(trace for trace in figure.data if trace.name == '每帧采集位置')
            self.assertEqual(frames.mode, 'markers'); self.assertIsNone(frames.text)
            np.testing.assert_array_equal(frames.customdata, [63, 64])
            trajectory = next(trace for trace in figure.data if trace.name == '行人 P7 中心轨迹')
            self.assertEqual(trajectory.mode, 'lines'); self.assertEqual(len(trajectory.x), 2)
            person_start = next(trace for trace in figure.data if trace.name == '行人 P7 起点')
            self.assertEqual(person_start.mode, 'markers'); self.assertEqual(len(person_start.x), 1)
            np.testing.assert_array_equal(person_start.customdata, [63])
            canvas.close()

    def test_map_is_a_result_mode_without_a_separate_tab_or_process_button(self):
        window = AnnotationWindow()
        self.assertEqual(window.result_tabs.tabText(1), '结果可视化')
        self.assertNotIn('全帧点云地图', [window.result_tabs.tabText(i) for i in range(window.result_tabs.count())])
        self.assertFalse(any('全帧处理' in b.text() for b in window.map_page.findChildren(QPushButton)))
        with patch.object(window, 'current_context', return_value={'output': '/tmp/map-test'}), patch.object(window, 'refresh_preview'), patch.object(window.map_page, 'refresh') as refresh:
            window.show_map_mode()
            self.assertEqual(window.preview_stack.currentWidget(), window.map_page)
            self.assertTrue(window.map_button.isChecked())
            self.assertFalse(window.frame_controls.isEnabled())
            self.assertTrue(window.frame_controls.isHidden())
            self.assertTrue(window.map_page.canvas.show_start)
            self.assertTrue(window.map_page.canvas.show_frames)
            self.assertNotIn('show_frame_ids', window.map_page.display_actions)
            window.map_page.display_actions['show_person_paths'].setChecked(False)
            self.assertFalse(window.map_page.canvas.show_person_paths)
            window.map_page.display_actions['show_person_paths'].setChecked(True)
            refresh.assert_called_once()
            window.show_label_mode('combined')
            self.assertEqual(window.preview_stack.currentWidget(), window.frame_view_panel)
            self.assertTrue(window.frame_controls.isEnabled())
            self.assertFalse(window.frame_controls.isHidden())
            self.assertTrue(window.combined_button.isChecked())
        window.close()

    def test_successful_optimization_automatically_generates_map_without_switching_views(self):
        window = AnnotationWindow(); window.selected_stages = ['optimize']
        with patch.object(window.map_page, 'refresh') as refresh, patch('tools.lidar_annotator.optimized_map.optimized_map_ready', return_value=True), patch.object(window, 'current_context', return_value={'output': '/tmp/map-test'}):
            window.finished(0, QProcess.ExitStatus.NormalExit)
        self.assertEqual(window.result_tabs.currentIndex(), 0)
        refresh.assert_called_once()
        window.close()


if __name__ == '__main__':unittest.main()
