"""Automatic user-config persistence without modifying bundled defaults."""
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QPushButton
from PIL import Image

from .recipe_config import RECIPE_PATH, default_recipe, load_recipe, save_system_recipe, system_recipe_path
from .prompt_debug_dialog import PromptDebugDialog
from .qt_app import AnnotationWindow


class RecipeAutosaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {'XDG_CONFIG_HOME': directory.name})
        environment.start(); self.addCleanup(environment.stop)

    def test_rule_edits_autosave_without_a_save_dialog_and_restart_restores(self):
        baseline = RECIPE_PATH.read_bytes()
        dialog = PromptDebugDialog(default_recipe())
        self.assertFalse(any('保存为标注配置' in b.text() for b in dialog.findChildren(QPushButton)))
        loop = QEventLoop()
        dialog.recipe_saved.connect(lambda _: loop.quit())
        timeout = QTimer(); timeout.setSingleShot(True); timeout.timeout.connect(loop.quit); timeout.start(1500)
        with patch('tools.lidar_annotator.prompt_debug_dialog.QFileDialog.getSaveFileName') as picker:
            dialog.rules_editor.tables['subtractions'].cellWidget(0, 0).setChecked(False)
            loop.exec(); timeout.stop()
            picker.assert_not_called()
        saved = load_recipe(system_recipe_path())
        self.assertFalse(saved['composition_rules']['subtractions'][0]['enabled'])
        self.assertIn('已自动保存', dialog.autosave_status.text())
        self.assertEqual(RECIPE_PATH.read_bytes(), baseline)
        dialog.close()
        window = AnnotationWindow()
        self.assertEqual(window.recipe_profile, str(system_recipe_path()))
        window.close()
        reopened = PromptDebugDialog(load_recipe(system_recipe_path()))
        self.assertFalse(reopened.rules_editor.tables['subtractions'].cellWidget(0, 0).isChecked())
        reopened.close()

    def test_close_flushes_committed_prompts_and_class_state(self):
        dialog = PromptDebugDialog(default_recipe())
        dialog.editor.table.cellWidget(0, 3).setPlainText('road\nsidewalk\npaved street')
        dialog.editor.table.cellWidget(2, 0).setChecked(False)
        dialog.refinements[('manhole_covers', 'prompt')].setPlainText('drain cover')
        dialog.close()
        saved = load_recipe(system_recipe_path())
        self.assertIn('paved street', saved['base_semantics']['prompts']['0'])
        self.assertIn(2, saved['disabled_classes'])
        self.assertEqual(saved['manhole_covers']['prompt'], 'drain cover')

    def test_invalid_edit_preserves_last_valid_system_config(self):
        path = save_system_recipe(default_recipe()); before = path.read_bytes()
        dialog = PromptDebugDialog(default_recipe())
        dialog.rules_editor.tables['subtractions'].cellWidget(0, 2).choices[3].setChecked(True)
        dialog.autosave_recipe()
        self.assertEqual(path.read_bytes(), before)
        self.assertIn('配置未保存', dialog.autosave_status.text())
        dialog.close()

    def test_debug_autosave_selects_system_recipe_for_main_lmdb_execution(self):
        window = AnnotationWindow(); window.input_format.setCurrentIndex(window.input_format.findData('lmdb'))
        path = Path(self._testMethodName + '.png')
        # Keep all files inside this test's isolated config root.
        path = system_recipe_path().parent.parent / path
        Image.new('RGB', (8, 6)).save(path)
        def edit_and_close(dialog):
            dialog.rules_editor.tables['subtractions'].cellWidget(0, 0).setChecked(False)
            dialog.close()
        context = dict(output=str(path.parent / 'job'), input_format='lmdb')
        with patch.object(window, 'current_context', return_value=context), patch('tools.lidar_annotator.live_preview.frame_inputs', return_value={'image': path}), patch.object(PromptDebugDialog, 'exec', edit_and_close):
            window.debug_prompts()
        self.assertEqual(window.profile.text(), str(system_recipe_path()))
        self.assertFalse(load_recipe(window.profile.text())['composition_rules']['subtractions'][0]['enabled'])
        window.close()


if __name__ == '__main__':
    unittest.main()
