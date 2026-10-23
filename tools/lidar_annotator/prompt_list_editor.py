"""Prompt chips: commit one phrase at a time instead of editing a text block."""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QLineEdit,
    QListView, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

class PromptInput(QLineEdit):
    cancelled = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit(); event.accept()
        else:
            super().keyPressEvent(event)


class PromptListEditor(QWidget):
    textChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._prompts = []
        layout = QVBoxLayout(self); layout.setContentsMargins(4, 3, 4, 3); layout.setSpacing(3)
        self.chips = QListWidget()
        self.chips.setViewMode(QListView.ViewMode.IconMode)
        self.chips.setFlow(QListView.Flow.LeftToRight)
        self.chips.setWrapping(True)
        self.chips.setResizeMode(QListView.ResizeMode.Adjust)
        self.chips.setMovement(QListView.Movement.Static)
        self.chips.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chips.setSpacing(3); self.chips.setMinimumHeight(30)
        self.chips.setStyleSheet('QListWidget {border:0;background:transparent;}')
        tags = QHBoxLayout(); tags.setSpacing(4)
        self.plus_button = QPushButton('＋'); self.plus_button.setAutoDefault(False)
        self.plus_button.setFixedSize(28, 28); self.plus_button.setToolTip('添加提示词')
        self.plus_button.clicked.connect(self.begin_add)
        tags.addWidget(self.chips, 1); tags.addWidget(self.plus_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(tags, 1)
        self.entry = QWidget()
        row = QHBoxLayout(self.entry); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
        self.input = PromptInput(); self.input.setPlaceholderText('输入提示词，回车添加；Esc 取消')
        self.input.cancelled.connect(self.cancel_add)
        self.add_button = QPushButton('添加'); self.add_button.setAutoDefault(False)
        self.add_button.clicked.connect(self.add_prompt); self.input.returnPressed.connect(self.add_prompt)
        row.addWidget(self.input, 1); row.addWidget(self.add_button); layout.addWidget(self.entry)
        self.entry.hide()

    def begin_add(self):
        self.entry.show(); self.plus_button.hide(); self.input.setFocus()

    def cancel_add(self):
        self.input.clear(); self.entry.hide(); self.plus_button.show()

    def toPlainText(self):
        return '\n'.join(self._prompts)

    def setPlainText(self, text):
        self._prompts = [p.strip() for p in text.splitlines() if p.strip()]
        self.cancel_add(); self._render(); self.textChanged.emit()

    def add_prompt(self):
        text = self.input.text().strip()
        if not text:
            return
        if text.casefold() in {p.casefold() for p in self._prompts}:
            self.input.setToolTip('这条提示词已经存在')
            return
        self._prompts.append(text); self.input.clear(); self.input.setToolTip('')
        self.cancel_add(); self._render(); self.textChanged.emit()

    def remove_prompt(self, index):
        del self._prompts[index]
        self._render(); self.textChanged.emit()

    def _render(self):
        self.chips.clear()
        for index, text in enumerate(self._prompts):
            chip = QWidget(); chip.setObjectName('promptChip')
            chip.setStyleSheet('QWidget#promptChip {background:#e8f1f8;border:1px solid #bfd4e5;border-radius:5px;}')
            row = QHBoxLayout(chip); row.setContentsMargins(7, 2, 3, 2); row.setSpacing(5)
            label = QLabel(text); label.setTextFormat(Qt.TextFormat.PlainText); row.addWidget(label)
            remove = QPushButton('×'); remove.setAutoDefault(False); remove.setFixedSize(20, 20)
            remove.setToolTip('删除提示词：' + text)
            remove.setStyleSheet('QPushButton {border:0;background:transparent;color:#526777;} QPushButton:hover {color:#b42318;}')
            remove.clicked.connect(lambda checked=False, i=index: self.remove_prompt(i)); row.addWidget(remove)
            item = QListWidgetItem(); item.setSizeHint(chip.sizeHint().expandedTo(QSize(0, 28)))
            self.chips.addItem(item); self.chips.setItemWidget(item, chip)
