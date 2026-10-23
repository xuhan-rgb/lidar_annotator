"""Editable ordered semantic rules for the postprocessing page."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
    QHeaderView, QHBoxLayout, QLabel, QPushButton, QMenu, QTableWidget,
    QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .composition_rules import composition_rules, default_composition_rules


class ClassSelection(QToolButton):
    changed = Signal()

    def __init__(self, names, selected, unknown=False):
        super().__init__()
        self.names = names
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self); self.setMenu(menu)
        self.choices = {}
        for cid in [*range(10), *([255] if unknown else [])]:
            action = QAction(self); action.setCheckable(True); action.setChecked(cid in selected)
            menu.addAction(action); self.choices[cid] = action
            action.toggled.connect(self.refresh)
        self.refresh()

    def selected(self):
        return [cid for cid, action in self.choices.items() if action.isChecked()]

    def refresh(self):
        for cid, action in self.choices.items():
            action.setText(f'{cid} · {self.names.get(cid, "未知 / ignore")}')
        text = ', '.join(str(cid) for cid in self.selected()) or '选择类别…'
        self.setText(text + ' ▾')
        self.setToolTip('\n'.join(self.choices[c].text() for c in self.selected()))
        self.changed.emit()


class CompositionRuleEditor(QTabWidget):
    changed = Signal()

    def __init__(self, recipe, parent=None):
        super().__init__(parent)
        self.recipe = recipe
        self.names = {c['id']: c['name'] for c in recipe['raw_classes']}
        self.tables = {}
        for kind, title, headers, explanation in (
            ('subtractions', '类别减法', ['启用', '原类别', '减去类别（多选）', '重叠区域归属'],
             '地面修正后按顺序扣除。使用达到门槛的类别候选区域；多个被减类别命中时取最高得分。屋顶候选仍须符合既有建筑与顶棚规则。'),
            ('priorities', '优先规则', ['启用', '优先类别', '覆盖类别（多选）', '使用证据', '生效条件', '比较类别（多选）', '最低像素得分'],
             '基础类别竞争后，从上到下执行优先规则，再做地面修正和类别减法。强实例证据沿用实例分数 ≥0.7 的支持；最低像素得分是当前加权证据得分。'),
        ):
            page = QWidget(); layout = QVBoxLayout(page)
            label = QLabel(explanation); label.setWordWrap(True); layout.addWidget(label)
            table = QTableWidget(0, len(headers)); table.setHorizontalHeaderLabels(headers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True); table.setAlternatingRowColors(True)
            table.setColumnWidth(0, 50); layout.addWidget(table, 1); self.tables[kind] = table
            actions = QHBoxLayout()
            for text, callback in (('＋ 添加规则', lambda _, k=kind: self.add_rule(k)),
                                   ('删除选中规则', lambda _, k=kind: self.remove_rule(k)),
                                   ('↑ 上移', lambda _, k=kind: self.move_rule(k, -1)),
                                   ('↓ 下移', lambda _, k=kind: self.move_rule(k, 1))):
                button = QPushButton(text); button.setAutoDefault(False); button.clicked.connect(callback); actions.addWidget(button)
            actions.addStretch()
            reset = QPushButton('恢复默认规则'); reset.setAutoDefault(False)
            reset.clicked.connect(lambda _, k=kind: self.restore_kind(k)); actions.addWidget(reset)
            layout.addLayout(actions); self.addTab(page, title)
        self.set_rules(composition_rules(recipe))

    def class_combo(self, selected):
        combo = QComboBox()
        for cid in range(10):combo.addItem(f'{cid} · {self.names[cid]}', cid)
        combo.setCurrentIndex(selected)
        return combo

    def combo(self, choices, selected):
        combo = QComboBox()
        for value, text in choices:combo.addItem(text, value)
        combo.setCurrentIndex(combo.findData(selected))
        return combo

    def set_rules(self, rules):
        for kind, table in self.tables.items():
            table.setRowCount(0)
            for rule in rules[kind]:
                row = table.rowCount(); table.insertRow(row); table.setRowHeight(row, 42)
                enabled = QCheckBox(); enabled.setChecked(rule['enabled'])
                source = self.class_combo(rule['source'])
                targets = ClassSelection(self.names, rule['subtract' if kind == 'subtractions' else 'targets'], kind == 'priorities')
                widgets = [enabled, source, targets]
                if kind == 'subtractions':
                    widgets.append(self.combo([('target', '归入被减类别'), ('ignore', '设为未知 / ignore')], rule['result']))
                else:
                    widgets.extend([
                        self.combo([('class', '类别候选'), ('strong', '强实例支持'), ('grass', '草地提示词')], rule['evidence']),
                        self.combo([('always', '命中即覆盖'), ('ge_current', '得分 ≥ 原类别'), ('gt_current', '得分 > 原类别'),
                                    ('ge_classes', '得分 ≥ 比较类别'), ('gt_classes', '得分 > 比较类别')], rule['condition']),
                        ClassSelection(self.names, rule['compare']),
                    ])
                    score = QDoubleSpinBox(); score.setRange(0, 1); score.setDecimals(3); score.setSingleStep(.05); score.setValue(rule['minimum_score']); widgets.append(score)
                    evidence_combo, condition_combo, compare_selector = widgets[3:6]
                    def adjust_evidence(index, source=source, evidence_combo=evidence_combo):
                        cid = source.currentData()
                        evidence_combo.model().item(1).setEnabled(cid in (1, 2))
                        evidence_combo.model().item(2).setEnabled(cid == 0)
                        if evidence_combo.currentData() == 'strong' and cid not in (1, 2) or evidence_combo.currentData() == 'grass' and cid != 0:
                            evidence_combo.setCurrentIndex(0)
                    source.currentIndexChanged.connect(adjust_evidence); adjust_evidence(0)
                    condition_combo.currentIndexChanged.connect(lambda _, c=condition_combo, w=compare_selector: w.setEnabled(c.currentData().endswith('_classes')))
                    compare_selector.setEnabled(condition_combo.currentData().endswith('_classes'))
                for column, widget in enumerate(widgets):
                    table.setCellWidget(row, column, widget)
                    if isinstance(widget, ClassSelection):widget.changed.connect(self.changed)
                    elif isinstance(widget, QCheckBox):widget.toggled.connect(self.changed)
                    elif isinstance(widget, QComboBox):widget.currentIndexChanged.connect(self.changed)
                    else:widget.valueChanged.connect(self.changed)
        self.changed.emit()

    def get_rules(self):
        result = {}
        for kind, table in self.tables.items():
            result[kind] = []
            for row in range(table.rowCount()):
                widgets = [table.cellWidget(row, c) for c in range(table.columnCount())]
                rule = dict(enabled=widgets[0].isChecked(), source=widgets[1].currentData())
                if kind == 'subtractions':
                    rule.update(subtract=widgets[2].selected(), result=widgets[3].currentData())
                else:
                    rule.update(targets=widgets[2].selected(), evidence=widgets[3].currentData(), condition=widgets[4].currentData(),
                                compare=widgets[5].selected(), minimum_score=widgets[6].value())
                result[kind].append(rule)
        return result

    def add_rule(self, kind):
        rules = self.get_rules()
        rule = dict(enabled=True, source=3)
        if kind == 'subtractions':rule.update(subtract=[9], result='target')
        else:rule.update(targets=[4], evidence='class', condition='always', compare=[], minimum_score=0.)
        rules[kind].append(rule); self.set_rules(rules)
        self.tables[kind].setCurrentCell(len(rules[kind]) - 1, 1)

    def remove_rule(self, kind):
        row = self.tables[kind].currentRow()
        if row < 0:return
        rules = self.get_rules(); del rules[kind][row]; self.set_rules(rules)
        if rules[kind]:self.tables[kind].setCurrentCell(min(row, len(rules[kind]) - 1), 1)

    def move_rule(self, kind, offset):
        row = self.tables[kind].currentRow(); destination = row + offset
        rules = self.get_rules()
        if row < 0 or not 0 <= destination < len(rules[kind]):return
        rules[kind].insert(destination, rules[kind].pop(row)); self.set_rules(rules)
        self.tables[kind].setCurrentCell(destination, 1)

    def restore_kind(self, kind):
        rules = self.get_rules(); rules[kind] = default_composition_rules(self.recipe)[kind]; self.set_rules(rules)

    def update_names(self, names):
        self.names.clear(); self.names.update(names)
        for table in self.tables.values():
            for row in range(table.rowCount()):
                source = table.cellWidget(row, 1)
                for cid in range(10):source.setItemText(cid, f'{cid} · {self.names[cid]}')
                for col in (2, 5) if table.columnCount() == 7 else (2,):table.cellWidget(row, col).refresh()
