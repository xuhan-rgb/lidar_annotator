"""Image-only prompt debugging: isolation, edits, command and result loading."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image
from PySide6.QtCore import QEventLoop, QProcess, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QHeaderView, QPushButton, QWidget

from .recipe_config import default_recipe, system_recipe_path
from .prompt_debug_dialog import PromptDebugDialog
from .sam_prompt_debug import prompt_thresholds, render_result
from .fresh_semantics import compose_image_semantics, _semantic_prompts


class PromptDebugTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {'XDG_CONFIG_HOME': directory.name})
        environment.start(); self.addCleanup(environment.stop)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_thresholds_include_refinements_and_respect_disabled_classes(self):
        recipe = default_recipe()
        values = prompt_thresholds(recipe)
        self.assertEqual(values['roof'], .4)
        self.assertEqual(values['metal canopy'], .65)
        self.assertEqual(values['manhole cover'], .7)
        self.assertEqual(values['small debris on the ground'], .45)
        recipe['disabled_classes'] = [0, 9]
        values = prompt_thresholds(recipe)
        self.assertNotIn('manhole cover', values)
        self.assertNotIn('roof', values)
        self.assertIn('person', values)

    def test_debug_window_has_no_transient_parent_and_moves_independently(self):
        main = QWidget()
        main.move(100, 120)
        dialog = PromptDebugDialog(default_recipe(), parent=main)
        self.assertIsNone(dialog.parentWidget())
        self.assertEqual(dialog.windowType(), Qt.WindowType.Window)
        position = main.pos()
        dialog.move(250, 280)
        self.assertEqual(main.pos(), position)
        dialog.close(); main.close()

    def test_editor_layout_separates_refinements_and_stretches_prompt_column(self):
        dialog = PromptDebugDialog(default_recipe())
        self.assertEqual([dialog.tabs.tabText(i) for i in range(dialog.tabs.count())], ['类别提示词', '后处理与合成规则', '原图 / 分割结果'])
        self.assertEqual(dialog.editor.table.horizontalHeader().sectionResizeMode(3), QHeaderView.ResizeMode.Stretch)
        for edit in dialog.refinements.values():
            self.assertTrue(dialog.refinement_page.isAncestorOf(edit))
            self.assertFalse(dialog.editor.isAncestorOf(edit))
        self.assertFalse(dialog.log.isVisible())
        dialog.set_running(True)
        self.assertFalse(dialog.refinement_page.isEnabled())
        dialog.set_running(False)
        self.assertTrue(dialog.refinement_page.isEnabled())
        dialog.close()

    def test_rules_are_only_in_postprocessing_and_saved_with_recipe(self):
        dialog = PromptDebugDialog(default_recipe())
        self.assertEqual(dialog.editor.table.columnCount(), 4)
        self.assertEqual([dialog.rules_editor.tabText(i) for i in range(3)], ['地面修正', '类别减法', '优先规则'])
        table = dialog.rules_editor.tables['subtractions']
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.cellWidget(0, 1).currentData(), 3)
        self.assertEqual(table.cellWidget(0, 2).selected(), [9])
        table.cellWidget(0, 0).setChecked(False)
        self.assertFalse(dialog.collect_recipe()['composition_rules']['subtractions'][0]['enabled'])
        dialog.close()

    def test_prompt_chips_commit_one_at_a_time_and_delete_individually(self):
        dialog = PromptDebugDialog(default_recipe())
        edit = dialog.editor.table.cellWidget(0, 3)
        self.assertEqual(edit.chips.count(), 2)
        edit.input.setText('paved street'); edit.input.returnPressed.emit()
        self.assertEqual(edit.toPlainText(), 'road\nsidewalk\npaved street')
        self.assertEqual(edit.input.text(), '')
        edit.input.setText('ROAD'); edit.add_button.click()
        self.assertEqual(edit.chips.count(), 3)
        chip = edit.chips.itemWidget(edit.chips.item(1))
        chip.findChild(QPushButton).click()
        self.assertEqual(edit.toPlainText(), 'road\npaved street')
        self.assertEqual(dialog.collect_recipe()['base_semantics']['prompts']['0'], ['road', 'paved street'])
        dialog.close()

    def test_plus_expands_input_and_commit_or_escape_collapses_it(self):
        dialog = PromptDebugDialog(default_recipe())
        edit = dialog.editor.table.cellWidget(0, 3)
        self.assertTrue(edit.entry.isHidden())
        self.assertFalse(edit.plus_button.isHidden())
        edit.plus_button.click()
        self.assertFalse(edit.entry.isHidden())
        edit.input.setText('paved street'); edit.input.returnPressed.emit()
        self.assertTrue(edit.entry.isHidden())
        self.assertIn('paved street', edit.toPlainText())
        edit.plus_button.click(); edit.input.setText('discard me')
        QTest.keyClick(edit.input, Qt.Key.Key_Escape)
        self.assertTrue(edit.entry.isHidden())
        self.assertEqual(edit.input.text(), '')
        self.assertNotIn('discard me', edit.toPlainText())
        dialog.close()

    def test_image_picker_opens_parsed_images_and_remembers_selected_folder(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        parsed = dialog.root / 'raw' / 'images'; parsed.mkdir(parents=True)
        dialog.image_directory = parsed
        chosen = dialog.root / 'other' / 'frame.png'; chosen.parent.mkdir()
        Image.new('RGB', (8, 6)).save(chosen)
        with patch('tools.lidar_annotator.prompt_debug_dialog.QFileDialog.getOpenFileName', return_value=(str(chosen), 'Images')) as picker:
            dialog.choose_image()
            self.assertEqual(picker.call_args.args[2], str(parsed))
        with patch('tools.lidar_annotator.prompt_debug_dialog.QFileDialog.getOpenFileName', return_value=('', '')) as picker:
            dialog.choose_image()
            self.assertEqual(picker.call_args.args[2], str(chosen.parent))
        dialog.close()

    def test_initial_image_uses_main_window_original_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '000007.jpg'
            Image.new('RGB', (20, 10), 'green').save(path)
            dialog = PromptDebugDialog(default_recipe(), image_path=path)
            self.assertEqual(dialog.image_path.text(), str(path))
            self.assertEqual(dialog.image_directory, path.parent)
            self.assertEqual(dialog.original_view.original.width(), 20)
            dialog.close()

    def test_main_window_passes_displayed_frame_and_original_image(self):
        from .qt_app import AnnotationWindow
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '000007.jpg'
            Image.new('RGB', (20, 10), 'green').save(path)
            window = AnnotationWindow()
            window.camera_view.data = {'frame': 7}
            with patch('tools.lidar_annotator.live_preview.frame_inputs', return_value={'image': path}) as inputs, patch('tools.lidar_annotator.prompt_debug_dialog.PromptDebugDialog') as dialog, patch('tools.lidar_annotator.qt_app.load_recipe', return_value=default_recipe()):
                window.debug_prompts()
                self.assertEqual(inputs.call_args.args[1], 7)
                self.assertEqual(dialog.call_args.kwargs['image_path'], path)
                self.assertEqual(dialog.call_args.args[1].size, (20, 10))
            window.camera_view.data = None
            window.close()

    def test_shared_composition_applies_building_intersection_roof_rule(self):
        recipe = default_recipe()
        evidence = {p: np.zeros((2, 2), np.float32) for ps in _semantic_prompts(recipe).values() for p in ps}
        evidence['building'][0, :] = .8
        evidence['roof'][:, 0] = .7
        labels, confidence = compose_image_semantics(evidence, np.zeros((2, 2, 2), np.float32), recipe)
        np.testing.assert_array_equal(labels, [[9, 3], [255, 255]])
        self.assertEqual(confidence[1, 0], 0)

    def test_render_preserves_original_pixels_outside_mask(self):
        rgb = np.full((2, 2, 3), 120, np.uint8)
        labels = np.array([[255, 1], [255, 255]], np.uint8)
        overlay, colors = render_result(Image.fromarray(rgb), labels, default_recipe())
        np.testing.assert_array_equal(np.array(overlay)[labels == 255], rgb[labels == 255])
        np.testing.assert_array_equal(np.array(colors)[0, 1], [0, 255, 255])

    def test_pipeline_command_freezes_edited_recipe_and_has_no_pointcloud_stage(self):
        original = default_recipe()
        dialog = PromptDebugDialog(original, Image.new('RGB', (8, 6)))
        dialog.editor.table.cellWidget(5, 3).setPlainText('glass railing')
        dialog.refinements[('manhole_covers', 'prompt')].setPlainText('drain cover')
        with patch.object(dialog.process, 'start') as start:
            dialog.start(False)
        args = start.call_args.args[1]
        self.assertIn('tools.lidar_annotator.sam_prompt_debug', args)
        self.assertNotIn('--prompt', args)
        self.assertNotIn('annotate', args)
        self.assertNotIn('--stages', args)
        snapshot = json.loads((dialog.output / 'recipe.json').read_text())
        self.assertEqual(snapshot['base_semantics']['prompts']['5'], ['glass railing'])
        self.assertEqual(snapshot['manhole_covers']['prompt'], 'drain cover')
        self.assertNotEqual(original['base_semantics']['prompts']['5'], ['glass railing'])
        self.assertTrue((dialog.output / 'input.png').is_file())
        dialog.close()

    def test_single_arbitrary_prompt_and_loading_saved_masks(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        dialog.prompt.setEditText('glass railing')
        with patch.object(dialog.process, 'start') as start:
            dialog.start(True)
        args = start.call_args.args[1]
        self.assertEqual(args[args.index('--prompt') + 1], 'glass railing')
        Image.new('RGB', (8, 6), 'magenta').save(dialog.output / 'prompt_000_overlay.png')
        Image.new('L', (8, 6), 255).save(dialog.output / 'prompt_000_mask.png')
        (dialog.output / 'report.json').write_text(json.dumps(dict(mode='single_prompt', prompts=[dict(prompt='glass railing', threshold=.25, scores=[.81], accepted=1, prefix='prompt_000')])))
        dialog.process_finished(0, QProcess.ExitStatus.NormalExit)
        self.assertEqual(dialog.result_choice.count(), 2)
        self.assertIn('0.810', dialog.details.text())
        self.assertFalse(dialog.result_view.original.isNull())
        self.assertTrue(dialog.pipeline_button.isEnabled())
        dialog.close()

    def test_repeat_timing_checkbox_only_applies_to_single_prompt(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        self.assertFalse(dialog.repeat_timing.isChecked())
        dialog.repeat_timing.setChecked(True)
        with patch.object(dialog.process, 'start') as start:
            dialog.start(True)
        self.assertIn('--repeat-timing', start.call_args.args[1])
        self.assertFalse(dialog.repeat_timing.isEnabled())
        dialog.set_running(False)
        self.assertTrue(dialog.repeat_timing.isChecked())
        with patch.object(dialog.process, 'start') as start:
            dialog.start(False)
        self.assertNotIn('--repeat-timing', start.call_args.args[1])
        dialog.close()

    def test_real_process_completion_loads_even_zero_detection_result(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        dialog.output = dialog.root
        Image.new('RGB', (8, 6)).save(dialog.output / 'p_overlay.png')
        Image.new('L', (8, 6)).save(dialog.output / 'p_mask.png')
        (dialog.output / 'report.json').write_text(json.dumps(dict(mode='single_prompt', prompts=[dict(prompt='road', threshold=.25, scores=[], accepted=0, prefix='p')])))
        dialog.set_running(True)
        loop = QEventLoop()
        dialog.process.finished.connect(lambda *_: QTimer.singleShot(0, loop.quit))
        timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); timer.start(3000)
        dialog.process.start(sys.executable, ['-c', 'print("complete", flush=True)'])
        loop.exec(); timer.stop()
        self.assertEqual(dialog.process.state(), QProcess.ProcessState.NotRunning)
        self.assertTrue(dialog.single_button.isEnabled())
        self.assertEqual(dialog.result_choice.count(), 2)
        self.assertFalse(dialog.result_view.original.isNull())
        self.assertIn('检出 0 个区域', dialog.details.text())
        dialog.close()

    def test_invalid_refinement_prevents_start(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        dialog.refinements[('ground_refinement', 'prompts')].setPlainText('pavement')
        with patch.object(dialog.process, 'start') as start:
            dialog.start(False)
        start.assert_not_called()
        self.assertIn('地面补全', dialog.status.text())
        dialog.close()

    def test_restore_and_save_include_all_image_pipeline_prompts(self):
        dialog = PromptDebugDialog(default_recipe(), Image.new('RGB', (8, 6)))
        dialog.refinements[('manhole_covers', 'prompt')].setPlainText('drain cover')
        button = next(b for b in dialog.editor.findChildren(QPushButton) if b.text() == '恢复全部默认')
        button.click()
        self.assertEqual(dialog.collect_recipe()['manhole_covers']['prompt'], 'manhole cover')
        target = system_recipe_path()
        with patch('tools.lidar_annotator.prompt_debug_dialog.QFileDialog.getSaveFileName') as picker:
            dialog.autosave_recipe()
        picker.assert_not_called()
        self.assertEqual(json.loads(target.read_text())['base_semantics'], default_recipe()['base_semantics'])
        self.assertIn('已自动保存', dialog.autosave_status.text())
        dialog.close()


if __name__ == '__main__':
    unittest.main()
