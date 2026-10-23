"""Focused checks for recipe validation and the offscreen Qt editor."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from .recipe_config import (class_prompts, default_recipe, load_recipe,
                            set_class_prompts, validate_recipe)
from .recipe_editor import RecipeDialog


class RecipeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_load_and_special_prompt_mappings(self):
        recipe = default_recipe()
        self.assertEqual(recipe, load_recipe())
        self.assertEqual(len(recipe["raw_classes"]), 10)
        self.assertEqual(class_prompts(recipe, 8), [recipe["ground_objects"]["prompt"]])
        self.assertEqual(class_prompts(recipe, 9), [recipe["roof_union"]["roof_prompt"], *recipe["roof_union"]["canopy_prompts"]])
        validate_recipe(recipe)

    def test_prompt_edits_preserve_fixed_metadata(self):
        recipe = default_recipe(); before = copy.deepcopy(recipe)
        set_class_prompts(recipe, 0, ["street", "walkway"])
        set_class_prompts(recipe, 8, ["loose object"])
        set_class_prompts(recipe, 9, ["roofing", "awning", "metal shade", "overhead shade"])
        validate_recipe(recipe)
        self.assertEqual(recipe["raw_to_training_image"], before["raw_to_training_image"])
        self.assertEqual(recipe["roof_union"]["roof_threshold"], before["roof_union"]["roof_threshold"])
        self.assertEqual(class_prompts(recipe, 0), ["street", "walkway"])

    def test_validation_rejects_bad_ids_empty_or_duplicate_shared_prompts(self):
        for mutate in (
            lambda r: r["raw_classes"].pop(),
            lambda r: r["raw_classes"][0].update(name=" "),
            lambda r: set_class_prompts(r, 0, ["", "sidewalk"]),
            lambda r: set_class_prompts(r, 1, ["road"]),
            lambda r: r.update(disabled_classes=[10]),
            lambda r: set_class_prompts(r, 9, ["roof"]),
        ):
            recipe = default_recipe()
            with self.assertRaises(ValueError):
                mutate(recipe)
                validate_recipe(recipe)

    def test_dialog_accept_edits_and_restore_buttons(self):
        baseline = default_recipe(); dialog = RecipeDialog(baseline)
        dialog.table.item(0, 2).setText("terrain")
        dialog.table.cellWidget(0, 3).setPlainText("street\nwalkway")
        dialog.table.cellWidget(1, 0).setChecked(False)
        dialog.buttons.button(dialog.buttons.StandardButton.Ok).click()
        result = dialog.get_recipe()
        self.assertEqual(result["raw_classes"][0]["name"], "terrain")
        self.assertIn(1, result["disabled_classes"])
        self.assertEqual(class_prompts(result, 0), ["street", "walkway"])

        dialog = RecipeDialog(baseline)
        dialog.table.item(0, 2).setText("temporary")
        dialog.table.item(1, 2).setText("preserve this")
        dialog.table.setCurrentCell(0, 2)
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "恢复当前类别默认值").click()
        self.assertEqual(dialog.table.item(0, 2).text(), baseline["raw_classes"][0]["name"])
        self.assertEqual(dialog.table.item(1, 2).text(), "preserve this")
        dialog.table.item(0, 2).setText("temporary")
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "恢复全部默认").click()
        self.assertEqual(dialog.table.item(0, 2).text(), baseline["raw_classes"][0]["name"])
        self.assertEqual(dialog.table.cellWidget(0, 3).toPlainText(), "\n".join(class_prompts(baseline, 0)))


if __name__ == "__main__":
    unittest.main()
