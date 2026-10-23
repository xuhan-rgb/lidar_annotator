"""Map page cache context follows optimized result file versions."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication

from .map_view import MapPage, MapReader


class MapRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_context(self, root):
        output = root / 'job'
        temporal = output / 'temporal'
        tracks = temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks'
        vehicles = temporal / 'map_review/history_fit/bev_dedup'
        source = root / 'external/source/index.json'
        pose = root / 'external/poses.npz'
        for path in (tracks / 'centers_only.npz', tracks / 'report.json', vehicles / 'vehicle_map_boxes.npz',
                     output / 'recipe.json', source, pose):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('version 1')
        return dict(output=str(output), backend=dict(source_index=str(source), pose_npz=str(pose), temporal_output=str(temporal)))

    def page(self, context):
        return MapPage(lambda: context)

    def begin_load(self, page):
        with patch.object(MapReader, 'start'):
            page.refresh()

    def test_unchanged_context_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.make_context(Path(directory)); page = self.page(context)
            self.begin_load(page)
            self.assertTrue(page.matches_context(context))
            page.deleteLater()

    def test_changed_saved_result_files_invalidate_context(self):
        changes = ('vehicle', 'pose', 'source_index', 'centers', 'report', 'recipe')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                context = self.make_context(Path(directory)); page = self.page(context)
                self.begin_load(page)
                if change == 'vehicle':
                    path = Path(context['backend']['temporal_output']) / 'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
                elif change == 'pose':
                    path = Path(context['backend']['pose_npz'])
                elif change == 'source_index':
                    path = Path(context['backend']['source_index'])
                elif change in ('centers', 'report'):
                    name = 'centers_only.npz' if change == 'centers' else 'report.json'
                    path = Path(context['backend']['temporal_output']) / 'map_review/sam3_full_video/interior_half_review/consistent_tracks' / name
                else:
                    path = Path(context['output']) / 'recipe.json'
                path.write_text('version 2 with a different size')
                self.assertFalse(page.matches_context(context))
                page.deleteLater()

    def test_changed_display_range_invalidates_context(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.make_context(Path(directory)); page = self.page(context)
            self.begin_load(page)
            changed = dict(context, map_limits=(10., -3., 6.))
            self.assertFalse(page.matches_context(changed))
            page.deleteLater()

    def test_result_change_during_load_rejects_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.make_context(Path(directory)); page = self.page(context)
            self.begin_load(page)
            Path(context['backend']['pose_npz']).write_text('new pose version')
            page.loaded({})
            self.assertIsNone(page.canvas.data)
            page.deleteLater()

