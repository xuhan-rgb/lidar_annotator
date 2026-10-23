"""Qt prompt editor and image-only SAM3 debug results."""
import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSplitter,
    QSizePolicy, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from .cli import REPO
from .recipe_config import default_recipe, validate_recipe, system_recipe_path, save_system_recipe
from .recipe_editor import RecipeDialog
from .composition_rule_editor import CompositionRuleEditor
from .composition_rules import default_composition_rules
from .prompt_list_editor import PromptListEditor
from .sam_prompt_debug import prompt_thresholds


class ImagePreview(QLabel):
    def __init__(self):
        super().__init__('尚无图像')
        self.original = QPixmap()
        self.setMinimumSize(160, 140)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('background:#14171c;color:white')

    def show_image(self, path):
        self.original = QPixmap(str(path))
        self.resizeEvent(None)

    def resizeEvent(self, event):
        if not self.original.isNull():
            self.setPixmap(self.original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class PromptChoice(QComboBox):
    def showPopup(self):
        dialog = self.window()
        try:
            choices = prompt_thresholds(dialog.collect_recipe())
            text = self.currentText()
            self.clear(); self.addItems(choices); self.setEditText(text)
        except (ValueError, KeyError) as exc:
            dialog.status.setText(f'配置无效：{exc}')
        super().showPopup()


class PromptDebugDialog(QDialog):
    recipe_saved = Signal(str)

    def __init__(self, recipe, image=None, parent=None, image_path=None):
        # A transient dialog can be moved together with its owner by the desktop.
        # Keep this as an independent native window even when an owner is supplied.
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle('SAM3 提示词调试 · 图像分割')
        self.resize(1100, 780)
        self.root = Path(tempfile.mkdtemp(prefix='sam3_prompts_', dir=tempfile.gettempdir()))
        self.image_directory = Path(image_path).parent if image_path else self.root
        self.output = None
        self.closing = False
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.image_path = QLineEdit()
        self.choose_button = QPushButton('选择原图…'); self.choose_button.clicked.connect(self.choose_image)
        row.addWidget(QLabel('测试图像')); row.addWidget(self.image_path, 1); row.addWidget(self.choose_button)
        layout.addLayout(row)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self.editor = RecipeDialog(recipe, self)
        self.editor.setWindowFlags(Qt.WindowType.Widget)
        self.editor.buttons.hide()
        table = self.editor.table
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['启用', 'ID', '类别名称', '提示词（＋ 添加）'])
        table.verticalHeader().hide()
        table.setAlternatingRowColors(True)
        table.setColumnWidth(0, 55); table.setColumnWidth(1, 45); table.setColumnWidth(2, 145)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        for cid in range(10):
            lines = len(table.cellWidget(cid, 3).toPlainText().splitlines())
            table.setRowHeight(cid, max(70, min(130, lines * 21 + 16)))
        self.tabs.addTab(self.editor, '类别提示词')
        ground_page = QWidget(); refinement_layout = QVBoxLayout(ground_page)
        hint = QLabel('这些提示词在基础类别分割后参与地面修正；运行整套提示词时自动应用。')
        hint.setWordWrap(True); refinement_layout.addWidget(hint)
        self.refinements = {}
        form = QFormLayout()
        for section, key, label in (('base_semantics', 'grass_prompts', '草地'), ('ground_refinement', 'prompts', '地面补全'), ('manhole_covers', 'prompt', '井盖'), ('small_debris', 'prompt', '碎屑')):
            edit = PromptListEditor(); edit.setFixedHeight(105 if key == 'prompts' else 85)
            value = recipe[section][key]
            edit.setPlainText('\n'.join(value) if isinstance(value, list) else value)
            self.refinements[(section, key)] = edit
            form.addRow(label, edit)
        refinement_layout.addLayout(form); refinement_layout.addStretch()
        restore = QPushButton('恢复后处理默认提示词'); restore.clicked.connect(self.restore_refinements)
        refinement_layout.addWidget(restore, alignment=Qt.AlignmentFlag.AlignLeft)
        for button in self.editor.findChildren(QPushButton):
            if button.text() == '恢复全部默认':
                button.clicked.connect(self.restore_postprocessing)
        self.rules_editor = CompositionRuleEditor(recipe)
        self.rules_editor.insertTab(0, ground_page, '地面修正')
        self.rules_editor.setCurrentIndex(0)
        self.refinement_page = self.rules_editor
        self.tabs.addTab(self.refinement_page, '后处理与合成规则')
        self.tabs.currentChanged.connect(self.update_rule_names)
        result = QWidget(); result_layout = QVBoxLayout(result)
        self.result_choice = QComboBox(); self.result_choice.currentIndexChanged.connect(self.show_result)
        self.result_choice.setPlaceholderText('运行后可切换最终结果及各条提示词的 mask')
        result_layout.addWidget(self.result_choice)
        views = QSplitter()
        self.original_view = ImagePreview(); self.result_view = ImagePreview()
        self.result_view.setText('尚未运行\n输入提示词后点击“测试单条”，或运行整套提示词')
        for title, view in (('原图', self.original_view), ('分割结果', self.result_view)):
            pane = QWidget(); pane_layout = QVBoxLayout(pane); pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.addWidget(QLabel(title)); pane_layout.addWidget(view, 1); views.addWidget(pane)
        views.setSizes([500, 500]); result_layout.addWidget(views, 1)
        self.details = QLabel(); self.details.setWordWrap(True); result_layout.addWidget(self.details)
        log_toggle = QPushButton('运行日志'); log_toggle.setCheckable(True)
        result_layout.addWidget(log_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); self.log.hide()
        log_toggle.toggled.connect(self.log.setVisible); result_layout.addWidget(self.log)
        self.result_tab = self.tabs.addTab(result, '原图 / 分割结果')
        test_group = QGroupBox('单条提示词测试'); test_layout = QHBoxLayout(test_group)
        row = QHBoxLayout()
        self.prompt = PromptChoice(self); self.prompt.setEditable(True); self.prompt.addItems(prompt_thresholds(recipe))
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(.01, 1.); self.threshold.setSingleStep(.05); self.threshold.setValue(.25)
        self.prompt.setToolTip('选择现有提示词，或直接输入任意新提示词；单条测试不会自动改变类别配置。')
        self.single_button = QPushButton('测试单条'); self.single_button.clicked.connect(lambda: self.start(True))
        self.repeat_timing = QCheckBox('测试时间（运行两次）')
        self.repeat_timing.setToolTip('仅用于单条测试：模型加载一次，连续执行两次图像特征提取和提示词解码，分别计时；结果使用第二次输出。')
        self.pipeline_button = QPushButton('运行整套提示词'); self.pipeline_button.clicked.connect(lambda: self.start(False))
        self.pipeline_button.setStyleSheet('QPushButton {background:#1769a6;color:white;padding:7px 14px;font-weight:600;} QPushButton:disabled {background:#a8b6c2;}')
        row.addWidget(self.prompt, 1); row.addWidget(QLabel('得分门槛')); row.addWidget(self.threshold); row.addWidget(self.repeat_timing); row.addWidget(self.single_button)
        test_layout.addLayout(row); layout.addWidget(test_group)
        row = QHBoxLayout()
        self.status = QLabel('修改类别或后处理提示词后，可运行整套图像语义流程。'); self.status.setWordWrap(True)
        self.autosave_status = QLabel('配置修改后自动保存')
        self.autosave_status.setWordWrap(True); self.autosave_status.setMaximumWidth(240)
        self.autosave_status.setToolTip(str(system_recipe_path()))
        self.stop_button = QPushButton('停止'); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.process.terminate)
        close = QPushButton('关闭'); close.clicked.connect(self.close)
        row.addWidget(self.status, 1); row.addWidget(self.autosave_status); row.addWidget(self.pipeline_button); row.addWidget(self.stop_button); row.addWidget(close); layout.addLayout(row)
        if image_path and Path(image_path).is_file():
            self.image_path.setText(str(image_path)); self.original_view.show_image(image_path)
        elif image is not None:
            path = self.root / 'input.png'; image.save(path); self.image_path.setText(str(path)); self.original_view.show_image(path)
        self.autosave_timer = QTimer(self); self.autosave_timer.setSingleShot(True); self.autosave_timer.setInterval(400)
        self.autosave_timer.timeout.connect(self.autosave_recipe)
        self._autosave_pending = False
        self._last_saved_recipe = None
        self.editor.table.itemChanged.connect(self.queue_autosave)
        for cid in range(10):
            self.editor.table.cellWidget(cid, 0).toggled.connect(self.queue_autosave)
            self.editor.table.cellWidget(cid, 3).textChanged.connect(self.queue_autosave)
        for edit in self.refinements.values():edit.textChanged.connect(self.queue_autosave)
        self.rules_editor.changed.connect(self.queue_autosave)

    def collect_recipe(self):
        recipe = self.editor.collect_recipe()
        for (section, key), edit in self.refinements.items():
            values = [p.strip() for p in edit.toPlainText().splitlines() if p.strip()]
            if not values or key == 'prompt' and len(values) != 1:
                raise ValueError('后处理提示词不能为空；井盖和碎屑各需一条提示词')
            if section == 'ground_refinement' and len(values) < recipe[section]['minimum_pixel_prompt_support']:
                raise ValueError('地面补全提示词数量不能少于共同支持数量')
            recipe[section][key] = values[0] if key == 'prompt' else values
        recipe['composition_rules'] = self.rules_editor.get_rules()
        validate_recipe(recipe)
        return recipe

    def update_rule_names(self, index):
        if index == 1:
            self.rules_editor.update_names({cid: self.editor.table.item(cid, 2).text() for cid in range(10)})

    def restore_postprocessing(self):
        self.restore_refinements()
        self.rules_editor.set_rules(default_composition_rules(default_recipe()))

    def restore_refinements(self):
        recipe = default_recipe()
        for (section, key), edit in self.refinements.items():
            value = recipe[section][key]
            edit.setPlainText('\n'.join(value) if isinstance(value, list) else value)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择测试原图', str(self.image_directory), 'Images (*.jpg *.jpeg *.png *.bmp)')
        if path:
            self.image_directory = Path(path).parent
            self.image_path.setText(path); self.original_view.show_image(path)

    def set_running(self, running):
        for widget in (self.editor, self.refinement_page, self.image_path, self.choose_button, self.prompt, self.threshold, self.repeat_timing, self.single_button, self.pipeline_button):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def start(self, single):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        try:
            recipe = self.collect_recipe()
            if self._autosave_pending:self.autosave_recipe()
            image = Path(self.image_path.text()).expanduser()
            if not image.is_file():
                raise ValueError('请先选择测试原图')
            text = self.prompt.currentText().strip()
            if single and not text:
                raise ValueError('请输入提示词')
            output = self.root / uuid4().hex[:12]; output.mkdir()
            recipe_path = output / 'recipe.json'
            recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
            # Freeze the tested input so editing/selecting another image cannot change this run.
            from PIL import Image
            Image.open(image).convert('RGB').save(output / 'input.png')
            self.original_view.show_image(output / 'input.png')
            args = ['-m', 'tools.lidar_annotator.sam_prompt_debug', '--image', str(output / 'input.png'), '--recipe', str(recipe_path), '--output', str(output)]
            if single:
                args += ['--prompt', text, '--threshold', str(self.threshold.value())]
                if self.repeat_timing.isChecked():args += ['--repeat-timing']
            self.output = output
            self.result_choice.clear(); self.result_view.clear(); self.result_view.original = QPixmap(); self.details.clear()
            self.log.clear(); self.tabs.setCurrentIndex(self.result_tab); self.status.setText('正在运行 SAM3 图像分割…'); self.result_view.setText('正在分割，请稍候…')
            self.set_running(True)
            self.process.setWorkingDirectory(str(REPO)); self.process.start(sys.executable, args)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f'无法启动：{exc}')

    def read_log(self):
        self.log.append(bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace'))

    def process_finished(self, code, status):
        self.read_log(); self.set_running(False)
        path = self.output / 'report.json' if self.output else None
        if code != 0 or path is None or not path.is_file():
            self.status.setText(f'调试未完成，请查看日志：{self.output}')
            self.result_view.setText('调试未完成，请查看下方日志'); self.log.show()
        else:
            report = json.loads(path.read_text())
            if report['mode'] == 'image_pipeline':
                self.result_choice.addItem('整套流程：最终语义叠加', ('final_overlay.png', '全部启用类别及地面、屋顶后处理合成结果'))
                self.result_choice.addItem('整套流程：最终类别颜色', ('final_colors.png', '未知区域显示为黑色'))
            for item in report['prompts']:
                details = f"{item['prompt']} · 门槛 {item['threshold']:g} · 检出 {item['accepted']} 个区域 · 分数 " + ', '.join(f'{s:.3f}' for s in item['scores'])
                if 'image_feature_ms' in report and 'decode_ms' in item:
                    details += f"\n图像特征提取 {report['image_feature_ms']:.2f} ms · decode {item['decode_ms']:.2f} ms（含文本编码与 mask 后处理）"
                if report.get('timing_runs'):
                    details += '\n' + '；'.join(f"第 {run['run']} 次：特征提取 {run['image_feature_ms']:.2f} ms / decode {run['decode_ms']:.2f} ms" for run in report['timing_runs'])
                self.result_choice.addItem(item['prompt'] + '：叠加', (item['prefix'] + '_overlay.png', details))
                self.result_choice.addItem(item['prompt'] + '：mask', (item['prefix'] + '_mask.png', details))
            self.result_choice.setCurrentIndex(0)
            self.show_result()
            self.status.setText(f'完成；调试结果：{self.output}')
        if self.closing:
            self.close()

    def show_result(self):
        value = self.result_choice.currentData()
        if value and self.output:
            self.result_view.show_image(self.output / value[0]); self.details.setText(value[1])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.set_running(False); self.status.setText('无法启动 SAM3：' + self.process.errorString())

    def queue_autosave(self, *args):
        self._autosave_pending = True
        self.autosave_status.setText('正在自动保存…')
        self.autosave_timer.start()

    def autosave_recipe(self):
        self.autosave_timer.stop()
        try:
            recipe = self.collect_recipe()
            if recipe != self._last_saved_recipe:
                path = save_system_recipe(recipe)
                self._last_saved_recipe = recipe
                self.recipe_saved.emit(str(path))
            self._autosave_pending = False
            self.autosave_status.setText('配置已自动保存')
        except (OSError, ValueError, KeyError) as exc:
            self.autosave_status.setText('配置未保存：' + str(exc))

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self._autosave_pending:self.autosave_recipe()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.closing = True; self.process.terminate(); event.ignore()
        else:
            event.accept()
