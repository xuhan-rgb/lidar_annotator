"""Offscreen payload checks; no live window or screenshot is opened."""
import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from .live_preview import load_frame,PointCanvas
from .test_track_preview import fixture


class CoveragePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_coverage_loads_raw_points_without_semantics_or_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            context=fixture(Path(tmp));data=load_frame(context,63,labels=False,mode='projection')
            self.assertFalse(data['semantic']);self.assertEqual(data['targets'],[]);self.assertEqual(data['records'],[])
            self.assertEqual(data['coverage']['bev_fov_mask'].shape,(160,160))
            self.assertIn('不代表完整标注或负样本',data['status'])
            self.assertEqual(len(data['palette']),32)
            canvas=PointCanvas(bev=True);canvas.set_frame(data)
            self.assertGreater(canvas.coverage_path.elementCount(),0)
            with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                canvas.paintEvent(None)
                painter.return_value.drawPath.assert_called_once_with(canvas.coverage_path)
                self.assertTrue(any(call.args[0].style()==Qt.PenStyle.DashLine and call.args[0].color().name()=='#d980fa' for call in painter.return_value.setPen.call_args_list if hasattr(call.args[0],'style')))
            canvas.show_coverage=False
            with patch('tools.lidar_annotator.live_preview.QPainter') as painter:
                canvas.paintEvent(None)
                painter.return_value.drawPath.assert_not_called()
            np.testing.assert_array_equal(data['bev_points'],np.array([[0,0],[.1,0],[1,0],[.5,0],[-1,0],[0,0]],np.float32))
