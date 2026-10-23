"""Offscreen checks for the raw camera coverage preview mode."""
import os,unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from .qt_app import AnnotationWindow
from .cli import DEFAULT


class CoverageButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_button_order_mode_selection_and_raw_preview(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertEqual([button.text() for button in window.mode_group.buttons()],[
                '查看点云投影','查看语义 mask','查看跟踪 ID','查看联合标注','查看全帧地图'])
            self.assertIn('不表示完整标注或负样本区域',window.projection_button.toolTip())
            context={'output':'/tmp/coverage-preview'}
            with patch.object(window,'current_context',return_value=context), \
                 patch.object(window,'projection_frame_context',return_value=context) as raw_context, \
                 patch.object(window,'refresh_preview') as refresh:
                window.preview_frame.setValue(12)
                window.projection_button.click()
                self.assertEqual(window.preview_frame.value(),12)
                self.assertEqual(window.preview_mode,'projection')
                self.assertFalse(window.preview_labels)
                self.assertTrue(window.projection_button.isChecked())
                self.assertEqual([key for key,button in window.mode_buttons.items() if button.isChecked()],['projection'])
                raw_context.assert_called_once_with(context,12)
                self.assertIn(((True,),{}),refresh.call_args_list)
                self.assertTrue(window.track_controls.isHidden())
                self.assertEqual(window.result_tabs.currentIndex(),1)
        finally:window.close()

    def test_range_toggle_only_in_projection_and_immediately_after_reset(self):
        window=AnnotationWindow(DEFAULT)
        try:
            self.assertTrue(window.coverage_toggle.isHidden())
            options=window.frame_view_panel.layout().itemAt(1).layout()
            index=options.indexOf(window.coverage_toggle)
            self.assertEqual(options.itemAt(index-1).widget().text(),'重置视图')
            self.assertNotIn(window.coverage_toggle,window.mode_group.buttons())
            context={'output':'/tmp/coverage-preview'}
            with patch.object(window,'current_context',return_value=context),patch.object(window,'projection_frame_context',return_value=context),patch.object(window,'refresh_preview'):
                window.projection_button.click()
                self.assertFalse(window.coverage_toggle.isHidden())
                window.coverage_toggle.click()
                self.assertFalse(window.bev_view.show_coverage)
                self.assertEqual(window.preview_mode,'projection')
                window.coverage_toggle.click()
                self.assertTrue(window.bev_view.show_coverage)
                for mode in ('semantic','tracks','combined'):
                    window.show_label_mode(mode)
                    self.assertTrue(window.coverage_toggle.isHidden())
                window.show_projection()
                self.assertFalse(window.coverage_toggle.isHidden())
                with patch.object(window.map_page,'matches_context',return_value=True):window.show_map_mode()
                self.assertTrue(window.coverage_toggle.isHidden())
        finally:window.close()

    def test_failed_coverage_load_restores_previous_mode(self):
        window=AnnotationWindow(DEFAULT)
        try:
            with patch.object(window,'current_context',side_effect=ValueError('invalid source')):
                window.projection_button.click()
            self.assertEqual(window.preview_mode,'combined')
            self.assertTrue(window.combined_button.isChecked())
            self.assertFalse(window.projection_button.isChecked())
        finally:window.close()


if __name__=='__main__':unittest.main()
