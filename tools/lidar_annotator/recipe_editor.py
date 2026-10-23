"""Small Qt editor for editable recipe names, prompts, and class enablement."""
from copy import deepcopy

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from .recipe_config import class_prompts, set_class_prompts, validate_recipe
from .prompt_list_editor import PromptListEditor


class RecipeDialog(QDialog):
    def __init__(self, recipe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM3 类别与提示词")
        self.resize(850, 650)
        self._recipe = deepcopy(recipe)
        self._defaults = None
        from .recipe_config import default_recipe
        self._defaults = default_recipe()
        self._build()
        self._populate(self._recipe)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("点击 ＋ 添加提示词，回车确认，点击 × 删除；屋顶类依次保留屋顶提示词和三个顶棚提示词。"))
        self.table = QTableWidget(10, 4, self)
        self.table.setHorizontalHeaderLabels(["启用", "ID", "类别名称", "提示词（＋ 添加）"])
        self.table.setColumnWidth(0, 60); self.table.setColumnWidth(1, 45)
        self.table.setColumnWidth(2, 150); self.table.setColumnWidth(3, 420)
        for cid in range(10):
            self.table.setRowHeight(cid, 90)
            check = QCheckBox(self.table); check.setChecked(True); self.table.setCellWidget(cid, 0, check)
            item = QTableWidgetItem(str(cid)); item.setFlags(item.flags() & ~item.flags().ItemIsEditable)
            self.table.setItem(cid, 1, item)
            self.table.setItem(cid, 2, QTableWidgetItem())
            edit = PromptListEditor(); edit.setMinimumHeight(66); self.table.setCellWidget(cid, 3, edit)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        current = QPushButton("恢复当前类别默认值", self); current.clicked.connect(self.restore_current_class)
        defaults = QPushButton("恢复全部默认", self); defaults.clicked.connect(lambda: self._populate(self._defaults))
        actions.addWidget(current); actions.addWidget(defaults); actions.addStretch(1); layout.addLayout(actions)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self._accept_recipe); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def restore_current_class(self):
        """Restore the selected row from the approved baseline recipe."""
        cid = self.table.currentRow()
        if cid < 0:
            return
        baseline = {entry["id"]: entry for entry in self._defaults["raw_classes"]}[cid]
        self.table.cellWidget(cid, 0).setChecked(cid not in self._defaults.get("disabled_classes", []))
        self.table.item(cid, 2).setText(baseline["name"])
        self.table.cellWidget(cid, 3).setPlainText("\n".join(class_prompts(self._defaults, cid)))

    def _populate(self, recipe):
        disabled = set(recipe.get("disabled_classes", []))
        by_id = {entry["id"]: entry for entry in recipe["raw_classes"]}
        for cid in range(10):
            self.table.cellWidget(cid, 0).setChecked(cid not in disabled)
            self.table.item(cid, 2).setText(by_id[cid]["name"])
            self.table.cellWidget(cid, 3).setPlainText("\n".join(class_prompts(recipe, cid)))

    def collect_recipe(self):
        """Validate the current edits without closing the editor."""
        updated = deepcopy(self._recipe)
        disabled = []
        by_id = {entry["id"]: entry for entry in updated["raw_classes"]}
        for cid in range(10):
            if not self.table.cellWidget(cid, 0).isChecked():
                disabled.append(cid)
            by_id[cid]["name"] = self.table.item(cid, 2).text().strip()
            set_class_prompts(updated, cid, self.table.cellWidget(cid, 3).toPlainText().splitlines())
        updated["disabled_classes"] = disabled
        validate_recipe(updated)
        return updated

    def _accept_recipe(self):
        try:
            updated = self.collect_recipe()
        except (ValueError, KeyError) as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        self._recipe = updated
        self.accept()

    def get_recipe(self):
        """Return a validated copy of the accepted recipe."""
        if self.result() != QDialog.DialogCode.Accepted:
            raise RuntimeError("recipe is available only after accepting the dialog")
        validate_recipe(self._recipe)
        return deepcopy(self._recipe)
