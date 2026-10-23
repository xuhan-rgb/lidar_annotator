"""Map cache behavior across reader instances and input changes."""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtWidgets import QApplication

from . import optimized_map
from .map_view import MapReader
from .test_optimized_map import fixture
from .recipe_config import default_recipe


class MapCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def read(self, context, spacing=.15):
        reader = MapReader(context, spacing)
        loaded, failed = [], []
        reader.loaded.connect(loaded.append)
        reader.failed.connect(failed.append)
        reader.run()
        self.assertFalse(failed, failed)
        self.assertEqual(len(loaded), 1)
        return loaded[0]

    def test_render_stage_generates_default_map_even_when_frame_render_is_cached(self):
        from .fresh_pipeline import stage
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            (root/'backend.json').write_text('{}')
            (root/'review').mkdir(exist_ok=True);(root/'review/index.html').write_text('review')
            with patch('tools.lidar_annotator.fresh_pipeline._stage') as render:
                stage(root,'render',1)
                render.assert_called_once()
            dest=root/'review/optimized_map'
            for name in ('map_points.npz','display_points.npz','map_3d.html','cache.json'):self.assertTrue((dest/name).is_file())
            stamps={p.name:p.stat().st_mtime_ns for p in dest.iterdir()}
            with patch('tools.lidar_annotator.fresh_pipeline._stage') as render:
                stage(root,'render',1)
                render.assert_not_called()
            self.assertEqual(stamps,{p.name:p.stat().st_mtime_ns for p in dest.iterdir()})
            (dest/'cache.json').unlink()
            with patch('tools.lidar_annotator.fresh_pipeline._stage') as render:
                stage(root,'render',1)
                render.assert_not_called()
            self.assertTrue((dest/'cache.json').is_file())

    def test_reopening_reader_reuses_map_and_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = fixture(root)
            first = self.read(context)
            point_path = Path(first['archive'])
            html_path = Path(first['html'])
            point_mtime = point_path.stat().st_mtime_ns
            html_mtime = html_path.stat().st_mtime_ns

            with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics, \
                 patch.object(optimized_map, 'voxel_indices', wraps=optimized_map.voxel_indices) as voxels, \
                 patch('plotly.graph_objects.Figure.write_html', autospec=True) as write_html:
                second = self.read(context)

            semantics.assert_not_called()
            voxels.assert_not_called()
            write_html.assert_not_called()
            self.assertTrue(second['cache_reused'])
            self.assertEqual(Path(second['archive']).stat().st_mtime_ns, point_mtime)
            self.assertEqual(html_path.stat().st_mtime_ns, html_mtime)
            np.testing.assert_array_equal(second['xyz'], first['xyz'])
            np.testing.assert_array_equal(second['labels'], first['labels'])
            self.assertEqual(second['html'], first['html'])

    def test_rewriting_identical_index_and_recipe_reuses_saved_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);context,_,_=fixture(root)
            recipe=root/'recipe.json';recipe.write_text(json.dumps(default_recipe()))
            self.read(context)
            for path in (root/'data/labels/zgc/index.json',recipe):path.write_text(path.read_text())
            with patch.object(optimized_map,'frame_semantics',wraps=optimized_map.frame_semantics) as semantics:
                data=self.read(context)
            semantics.assert_not_called();self.assertTrue(data['cache_reused'])

    def test_spacing_change_resamples_saved_points_without_reading_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = fixture(root)
            first = self.read(context)
            point_path = Path(first['archive'])
            point_mtime = point_path.stat().st_mtime_ns
            with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics:
                second = self.read(context, .5)
            semantics.assert_not_called()
            self.assertEqual(point_path.stat().st_mtime_ns, point_mtime)
            self.assertEqual(second['spacing'], .5)
            self.assertTrue(second['cache_reused'])
            report = json.loads((root / 'review/optimized_map/report.json').read_text())
            self.assertEqual(report['display_voxel_m'], .5)
            self.assertEqual(report['display_points'], len(second['xyz']))

    def test_changed_inputs_and_display_range_invalidate_cached_result(self):
        mutations = (
            lambda root, context: self._change_frame_npz(root),
            lambda root, context: self._change_recipe(root),
            lambda root, context: self._change_centres(root),
            lambda root, context: self._change_pose(root),
            lambda root, context: context.update(map_limits=(10., -3., 6.)),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context, _, _ = fixture(root)
                self.read(context)
                mutate(root, context)
                with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics:
                    result = self.read(context)
                semantics.assert_called()
                self.assertFalse(result['cache_reused'])
                if mutate is mutations[0]:
                    self.assertTrue(np.any(np.all(np.isclose(result['xyz'], [.25, 0, 5]), axis=1)))
                elif mutate is mutations[1]:
                    self.assertEqual(result['palette'][0]['name'], default_recipe()['raw_classes'][0]['name'] + ' changed')
                elif mutate is mutations[2]:
                    np.testing.assert_allclose(result['person_xyz'][0], [.25, 0, 5])
                elif mutate is mutations[3]:
                    np.testing.assert_allclose(result['trajectory'][1], [10.25, 2, 0])

    def test_image_mask_and_vehicle_box_changes_invalidate_and_update_results(self):
        for change in ('mask', 'vehicle'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context, _, _ = fixture(root)
                if change == 'mask':
                    label_path = root / 'data/labels/zgc/frames/000000.npz'
                    with np.load(label_path) as source:
                        arrays = {key: source[key] for key in source.files if key != 'point_raw_labels'}
                    np.savez_compressed(label_path, **arrays)
                first = self.read(context)
                if change == 'mask':
                    from PIL import Image
                    path = root / 'data/labels/zgc/image_masks/000000.png'
                    with Image.open(path) as image:
                        mask = np.asarray(image).copy()
                    mask[6, 4] = 9
                    Image.fromarray(mask).save(path)
                else:
                    path = root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
                    with np.load(path) as source:
                        boxes = source['box_3d_map'].copy()
                        ids = source['track_ids'].copy()
                    boxes[0, 0] += .5
                    np.savez_compressed(path, box_3d_map=boxes, track_ids=ids)
                with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics:
                    result = self.read(context)
                semantics.assert_called()
                self.assertFalse(result['cache_reused'])
                if change == 'mask':
                    self.assertFalse(np.array_equal(result['labels'], first['labels']))
                else:
                    self.assertAlmostEqual(result['boxes'][0, 0], first['boxes'][0, 0] + .5)

    def test_display_cache_recovers_from_archive_and_missing_html_only_reexports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, _, _ = fixture(root)
            first = self.read(context)
            display = root / 'review/optimized_map/display_points.npz'
            display.write_bytes(b'invalid display cache')
            with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics, \
                 patch.object(optimized_map, 'voxel_indices', wraps=optimized_map.voxel_indices) as voxels:
                recovered = self.read(context)
            semantics.assert_not_called()
            voxels.assert_called()
            self.assertTrue(recovered['cache_reused'])
            np.testing.assert_array_equal(recovered['xyz'], first['xyz'])
            with np.load(display) as saved:
                self.assertEqual(len(saved['xyz']), len(recovered['xyz']))

            Path(recovered['html']).unlink()
            with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics, \
                 patch.object(optimized_map, 'voxel_indices', wraps=optimized_map.voxel_indices) as voxels:
                exported = self.read(context)
            semantics.assert_not_called()
            voxels.assert_not_called()
            self.assertTrue(exported['cache_reused'])
            self.assertTrue(Path(exported['html']).is_file())

    def test_missing_or_invalid_cache_receipt_rebuilds_safely(self):
        for receipt_state in ('missing', 'invalid'):
            with self.subTest(receipt_state=receipt_state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context, _, _ = fixture(root)
                first = self.read(context)
                receipt = root / 'review/optimized_map/cache.json'
                if receipt_state == 'missing':
                    receipt.unlink()
                else:
                    receipt.write_text('{invalid json')
                with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics:
                    result = self.read(context)
                semantics.assert_called()
                self.assertFalse(result['cache_reused'])
                self.assertTrue(receipt.is_file())
                np.testing.assert_array_equal(result['xyz'], first['xyz'])

    def test_missing_or_corrupt_cached_points_are_rebuilt(self):
        for damage in ('missing', 'corrupt'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context, _, _ = fixture(root)
                first = self.read(context)
                archive = Path(first['archive'])
                if damage == 'missing':
                    archive.unlink()
                else:
                    archive.write_bytes(b'invalid npz')
                with patch.object(optimized_map, 'frame_semantics', wraps=optimized_map.frame_semantics) as semantics:
                    result = self.read(context)
                semantics.assert_called()
                self.assertFalse(result['cache_reused'])
                with np.load(result['archive']) as saved:
                    self.assertEqual(len(saved['xyz_map']), result['point_count'])

    @staticmethod
    def _change_frame_npz(root):
        path = root / 'data/labels/zgc/frames/000000.npz'
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['xyz'] = arrays['xyz'].copy()
        arrays['xyz'][0, 0] += .25
        np.savez_compressed(path, **arrays)

    @staticmethod
    def _change_recipe(root):
        recipe = default_recipe()
        recipe['raw_classes'][0]['name'] += ' changed'
        (root / 'recipe.json').write_text(json.dumps(recipe))

    @staticmethod
    def _change_centres(root):
        path = root / 'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['corrected_point_lidar'] = arrays['corrected_point_lidar'].astype(float)
        arrays['corrected_point_lidar'][0, 0] += .25
        np.savez_compressed(path, **arrays)

    @staticmethod
    def _change_pose(root):
        path = root / 'temporal/poses.npz'
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['transforms'] = arrays['transforms'].copy()
        arrays['transforms'][1, 0, 3] += .25
        np.savez_compressed(path, **arrays)


if __name__ == '__main__':
    unittest.main()
