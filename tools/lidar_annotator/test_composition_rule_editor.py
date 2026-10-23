"""Offscreen checks of rule editing, ordering, validation and persistence."""
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QPushButton

from .recipe_config import default_recipe, load_recipe, system_recipe_path
from .composition_rules import composition_rules
from .composition_rule_editor import CompositionRuleEditor
from .prompt_debug_dialog import PromptDebugDialog


class CompositionRuleEditorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {'XDG_CONFIG_HOME': directory.name})
        environment.start(); self.addCleanup(environment.stop)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_roundtrip_add_move_remove_and_restore(self):
        recipe = default_recipe(); editor = CompositionRuleEditor(recipe)
        self.assertEqual(editor.get_rules(), composition_rules(recipe))
        editor.add_rule('priorities')
        self.assertEqual(editor.get_rules()['priorities'][-1]['source'], 3)
        editor.move_rule('priorities', -1)
        self.assertEqual(editor.get_rules()['priorities'][-2]['source'], 3)
        editor.remove_rule('priorities')
        self.assertEqual(editor.get_rules(), composition_rules(recipe))
        editor.add_rule('subtractions'); editor.restore_kind('subtractions')
        self.assertEqual(editor.get_rules(), composition_rules(recipe))
        editor.close()

    def test_multiple_subtraction_targets_modes_and_name_updates(self):
        editor = CompositionRuleEditor(default_recipe())
        table = editor.tables['subtractions']
        table.cellWidget(0, 2).choices[4].setChecked(True)
        table.cellWidget(0, 3).setCurrentIndex(1)
        self.assertEqual(editor.get_rules()['subtractions'][0]['subtract'], [4, 9])
        self.assertEqual(editor.get_rules()['subtractions'][0]['result'], 'ignore')
        names = dict(editor.names); names[4] = '树冠'; editor.update_names(names)
        self.assertIn('树冠', table.cellWidget(0, 2).choices[4].text())
        editor.close()

    def test_priority_edits_validation_and_config_file_roundtrip(self):
        dialog = PromptDebugDialog(default_recipe())
        table = dialog.rules_editor.tables['priorities']
        table.cellWidget(0, 4).setCurrentIndex(0)
        table.cellWidget(0, 6).setValue(.45)
        expected = dialog.collect_recipe()
        self.assertEqual(expected['composition_rules']['priorities'][0]['condition'], 'always')
        path = system_recipe_path()
        dialog.autosave_recipe()
        saved = load_recipe(path)
        self.assertEqual(saved, expected)
        reopened = PromptDebugDialog(saved)
        self.assertEqual(reopened.collect_recipe(), expected)
        # Invalid self-subtraction is refused before model execution.
        selector = dialog.rules_editor.tables['subtractions'].cellWidget(0, 2)
        selector.choices[3].setChecked(True)
        with self.assertRaisesRegex(ValueError, '减法'):
            dialog.collect_recipe()
        reopened.close(); dialog.close()

    def test_restore_all_resets_rules_and_ground_prompts(self):
        dialog = PromptDebugDialog(default_recipe())
        dialog.rules_editor.tables['subtractions'].cellWidget(0, 0).setChecked(False)
        dialog.refinements[('manhole_covers', 'prompt')].setPlainText('drain cover')
        next(b for b in dialog.editor.findChildren(QPushButton) if b.text() == '恢复全部默认').click()
        recipe = dialog.collect_recipe()
        self.assertEqual(recipe['composition_rules'], composition_rules(default_recipe()))
        self.assertEqual(recipe['manhole_covers']['prompt'], 'manhole cover')
        dialog.close()


if __name__ == '__main__':
    unittest.main()
