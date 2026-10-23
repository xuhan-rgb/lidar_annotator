"""Qt desktop controls for the same annotation CLI; no separate algorithm path."""
import argparse,bisect,html,json,os,shlex,sys
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from PySide6.QtCore import QEvent,QProcess,QSettings,QTimer,Qt,QUrl
from PySide6.QtGui import QDesktopServices,QFontDatabase,QFont,QIcon
from PySide6.QtWidgets import (QApplication,QAbstractSpinBox,QButtonGroup,QCheckBox,QComboBox,QDialog,QPlainTextEdit,QDoubleSpinBox,QFileDialog,QFormLayout,QGroupBox,
    QHBoxLayout,QLabel,QLineEdit,QMainWindow,QProgressBar,QPushButton,QRadioButton,
    QTextBrowser,QTreeWidget,QTreeWidgetItem,QScrollArea,QSizePolicy,QSlider,QSpinBox,QSplitter,QStackedWidget,QStyle,QStyleOptionButton,QTabWidget,QVBoxLayout,QWidget)
from .cli import DEFAULT,REPO,STAGES,load_config
from .output_locations import visualization_location
from .frame_selection import parse_frames,debug_output,temporary_debug_output
from .recipe_config import RECIPE_PATH,load_recipe,system_recipe_path
from .live_preview import PointCanvas,FrameReader,fingerprint,semantic_legend_html
from .track_preview import track_view
from .result_runs import latest_result,remember_result
from .stage_timing import stage_timing_text
from .frame_history import FrameHistoryBox,test_history,record_test,finish_test
from .reset_results import clear_previous_results
from .job_paths import work_directory
from .storage import sample_path,default_source,default_output
from tools.autolabel.resource_paths import font_path

STAGE_NAMES=dict(semantic='语义标注与目标跟踪',instances='行人 / 车辆实例标签',temporal='位姿准备与实例关联',tracking='读取轨迹与行人选点',optimize='位置优化与车辆去重',validate='标签与轨迹校验',render='逐帧联合可视化',catalog='保留视觉 / 点云联合索引')
TEMPORAL_FLOW=('temporal','tracking','optimize','validate')
EXPORT_FLOW=('render','catalog')
ICON_PATH=REPO/'assets/icons/lidar-annotator.svg'


def command_for(profile,stages,workers,frames='',dry_run=False,refresh_semantic=False,refresh_instances=False):
    selected=[s for s in STAGES if s in stages]
    if not selected:raise ValueError('至少选择一个处理阶段')
    frame_ids=parse_frames(frames) or []
    if any(f<0 for f in frame_ids):raise ValueError('帧号必须大于等于 0')
    if frame_ids and selected!=['render']:raise ValueError('指定帧号时请只选择“逐帧联合可视化”，跟踪和优化需要完整序列')
    args=[str(REPO/'lidar_annotator.py'),'run','--config',str((REPO/profile).resolve()),'--quiet-report','--stages',*selected,'--workers',str(workers)]
    if frame_ids:args+=['--frames',*map(str,frame_ids)]
    if dry_run:args+=['--dry-run']
    if refresh_semantic:args+=['--refresh-semantic']
    if refresh_instances:args+=['--refresh-instances']
    return args


def lmdb_command(source,output,frames='',preview=True,dry_run=False):
    if not source.strip():raise ValueError('请选择 LMDB 采集根目录')
    if not output.strip():raise ValueError('请选择 LMDB 导出目录')
    def absolute(value):
        p=Path(value).expanduser();return str(p if p.is_absolute() else REPO/p)
    args=[str(REPO/'lidar_annotator.py'),'lmdb','--source',absolute(source),'--output',absolute(output)]
    if frames.strip():
        ids=parse_frames(frames)
        if any(i<0 for i in ids):raise ValueError('帧号必须大于等于 0')
        args+=['--frames',*map(str,ids)]
    if preview:args+=['--preview']
    if dry_run:args+=['--dry-run']
    return args


def annotate_command(source,output,stages,workers,frames='',dry_run=False,refresh_semantic=False,refresh_instances=False,recipe_path=None,semantic_classes=None,input_format=None,camera=None,dataset_output=None,refresh_postprocess=False):
    if not source.strip():raise ValueError('请选择 LMDB 采集根目录')
    if not output.strip():raise ValueError('请选择 LMDB 导出目录')
    selected=[s for s in STAGES if s in stages]
    if not selected:raise ValueError('至少选择一个处理阶段')
    def absolute(value):
        p=Path(value).expanduser();return str(p if p.is_absolute() else REPO/p)
    args=[str(REPO/'lidar_annotator.py'),'annotate','--source',absolute(source),'--output',absolute(dataset_output or output),'--quiet-report','--stages',*selected,'--workers',str(workers)]
    if frames.strip():
        ids=parse_frames(frames)
        if any(i<0 for i in ids):raise ValueError('帧号必须大于等于 0')
        args+=['--frames',*map(str,ids)]
    if dry_run:args+=['--dry-run']
    if refresh_semantic:args+=['--refresh-semantic']
    if refresh_instances:args+=['--refresh-instances']
    if refresh_postprocess:args+=['--refresh-postprocess']
    if recipe_path:args+=['--recipe',absolute(str(recipe_path))]
    if semantic_classes is not None:args+=['--semantic-classes',*map(str,semantic_classes)]
    if dataset_output:args+=['--work-dir',absolute(output)]
    if input_format=='kitti':
        args+=['--input-format','kitti']
        if camera:args+=['--camera',camera]
    return args




class StageCheckBox(QCheckBox):
    """Keep algorithm notes inline whenever the available text width permits."""
    def __init__(self,title,note='',parent=None):
        super().__init__(title+note,parent);self.title=title;self.note=note;self.setToolTip(title+note);self.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Fixed)

    def minimumSizeHint(self):
        size=super().minimumSizeHint();size.setWidth(0);return size

    def fit_text(self):
        option=QStyleOptionButton();self.initStyleOption(option)
        width=self.style().subElementRect(QStyle.SubElement.SE_CheckBoxContents,option,self).width()
        text=self.title+self.note
        if self.note and self.fontMetrics().horizontalAdvance(text)>width:
            text='\n'.join(self.fontMetrics().elidedText(line,Qt.TextElideMode.ElideRight,max(0,width)) for line in (self.title,self.note))
        if text!=self.text():self.setText(text);self.updateGeometry()

    def resizeEvent(self,event):
        super().resizeEvent(event);self.fit_text()


class FrameSpinBox(QSpinBox):
    """Navigate frames in reading order: up is previous, down is next."""
    def stepBy(self,steps):
        ids=getattr(self,'frame_ids',None)
        if ids:
            position=bisect.bisect_left(ids,self.value())
            self.setValue(ids[max(0,min(len(ids)-1,position-steps))])
        else:super().stepBy(-steps)

    def stepEnabled(self):
        flags=QAbstractSpinBox.StepEnabledFlag.StepNone
        if self.value()>self.minimum():flags|=QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
        if self.value()<self.maximum():flags|=QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        return flags


class FrameSlider(QSlider):
    def wheelEvent(self,event):
        delta=event.angleDelta().y() or event.angleDelta().x() or event.pixelDelta().y()
        if delta:self.setValue(self.value()-(max(1,abs(delta)//120) if delta>0 else -max(1,abs(delta)//120)))
        event.accept()


class TaskLog(QTextBrowser):
    """Plain worker output with explicit local visualization links."""
    def __init__(self):
        super().__init__();self.setOpenLinks(False);self.setReadOnly(True);self.document().setMaximumBlockCount(20000);self.pending='';self.directories=set()

    def clear(self):
        super().clear();self.pending='';self.directories=set()

    def appendPlainText(self,text):
        self.append('<div>'+html.escape(text).replace('\n','<br>')+'</div>')

    def append_directory(self,path,label='可视化目录'):
        path=str(path)
        if path in self.directories:return
        self.directories.add(path);url=QUrl.fromLocalFile(path).toString()
        self.append(html.escape(label)+'：<a href="'+html.escape(url,quote=True)+'">'+html.escape(path)+'</a>（点击打开）')

    def append_stream(self,text,final=False):
        parts=(self.pending+text).split('\n');self.pending='' if final else parts.pop()
        for line in parts:
            if line.startswith('标准数据目录：'):self.append_directory(line[len('标准数据目录：'):],label='标准数据目录')
            elif line.startswith('可视化目录：'):self.append_directory(line[len('可视化目录：'):])
            elif line:self.appendPlainText(line)


class AnnotationWindow(QMainWindow):
    def __init__(self,profile=DEFAULT):
        super().__init__();self.setWindowTitle('LiDAR Annotator · 激光点云标注工具');self.setWindowIcon(QIcon(str(ICON_PATH)));self.resize(1280,850)
        self.process=QProcess(self);self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.started.connect(self.record_test_start)
        self.process.readyReadStandardOutput.connect(self.read_stdout);self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.process_error)
        self.timer=QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.read_stage_log)
        self.offsets={};self.stopping=False;self.close_pending=False;self.active_config=None;self.selected_stages=[];self.result_contexts={}
        central=QWidget();layout=QVBoxLayout(central);self.setCentralWidget(central)
        title=QLabel('LiDAR Annotator  |  激光点云语义、行人与车辆标注');title.setStyleSheet('font-size:20px;font-weight:600');layout.addWidget(title)
        splitter=QSplitter();layout.addWidget(splitter,1);self.sidebar=QWidget();left=QVBoxLayout(self.sidebar)
        self.sidebar_scroll=QScrollArea();self.sidebar_scroll.setWidgetResizable(True);self.sidebar_scroll.setMinimumWidth(330);self.sidebar_scroll.setMaximumWidth(440);self.sidebar_scroll.setWidget(self.sidebar);splitter.addWidget(self.sidebar_scroll)
        self.source_group=QGroupBox('数据来源与配置');source_layout=QVBoxLayout(self.source_group);left.addWidget(self.source_group)
        self.input_format=QComboBox();self.input_format.addItem('LMDB 采集目录（index + shards）','lmdb');self.input_format.addItem('KITTI 序列（图像 / 点云 / 标定）','kitti');source_layout.addWidget(QLabel('数据格式'));source_layout.addWidget(self.input_format)
        self.profile_label=QLabel('处理配置');source_layout.addWidget(self.profile_label);row=QHBoxLayout();self.profile=QLineEdit(os.path.relpath(REPO/profile,REPO));self.profile_browse=QPushButton('选择配置…');self.profile_browse.clicked.connect(self.browse_profile);row.addWidget(self.profile);row.addWidget(self.profile_browse);source_layout.addLayout(row);self.cached_profile=self.profile.text();self.recipe_profile=str(system_recipe_path()) if system_recipe_path().is_file() else str(REPO/'configs/user_recipe.json') if (REPO/'configs/user_recipe.json').exists() else str(RECIPE_PATH);self._profile_mode='configured';self.recipe_button=QPushButton('编辑类别与提示词…');self.recipe_button.clicked.connect(self.debug_prompts);source_layout.addWidget(self.recipe_button)
        self.source_label=QLabel('LMDB 采集根目录');source_layout.addWidget(self.source_label);row=QHBoxLayout();self.lmdb_root=QLineEdit(str(default_source('lmdb')));self.lmdb_root.setPlaceholderText('包含 index.lmdb 和 shards 的目录');self.lmdb_browse=QPushButton('选择目录…');self.lmdb_browse.clicked.connect(self.browse_lmdb);row.addWidget(self.lmdb_root);row.addWidget(self.lmdb_browse);source_layout.addLayout(row)
        self.output_label=QLabel('LMDB 导出目录');source_layout.addWidget(self.output_label);self.lmdb_output=QLineEdit(str(default_output('lmdb')));source_layout.addWidget(self.lmdb_output);self.projection_button=QPushButton('查看点云投影');self.projection_button.clicked.connect(lambda:self.show_projection())
        self.camera_label=QLabel('相机通道');self.camera_channel=QComboBox();source_layout.addWidget(self.camera_label);source_layout.addWidget(self.camera_channel);self.camera_channel.currentIndexChanged.connect(self.camera_changed)
        self.source_paths={'lmdb':self.lmdb_root.text(),'kitti':str(default_source('kitti'))};self.output_paths={'lmdb':self.lmdb_output.text(),'kitti':str(default_output('kitti'))};self._source_mode='configured'
        self.summary=QLabel();self.summary.setWordWrap(True);self.profile.editingFinished.connect(self.update_summary)
        group=QGroupBox('处理流程');stages_layout=QVBoxLayout(group);self.checks={};self.temporal_flow=QCheckBox('行人 / 车辆时序优化',group);self.temporal_flow.hide();self.temporal_flow.toggled.connect(self.toggle_temporal_flow);self.output_group=QGroupBox('输出结果');output_layout=QVBoxLayout(self.output_group);self.export_flow=QCheckBox('生成标签与可视化',self.output_group);self.export_flow.setChecked(True);self.export_flow.setToolTip('固定保留语义、mask、置信度、检测、轨迹和点云索引，同时生成逐帧联合审核图。');self.export_flow.toggled.connect(self.toggle_export_flow);self.export_flow.hide();self.export_flow_label=QLabel('生成标签与可视化');self.export_flow_label.setWordWrap(True);self.export_flow_label.setToolTip(self.export_flow.toolTip());self.export_flow.toggled.connect(self.export_flow_label.setEnabled);output_layout.addWidget(self.export_flow_label)
        for stage in STAGES:
            note='（SAM3）' if stage=='semantic' else '（YOLO26x、WildDet3D；复用 SAM mask）' if stage=='instances' else ''
            check=StageCheckBox(STAGE_NAMES[stage],note,group);check.setChecked(stage in ('render','catalog'));self.checks[stage]=check
            check.hide()
            if stage=='temporal':
                label=QLabel(self.temporal_flow.text());label.setWordWrap(True);stages_layout.addWidget(label);self.temporal_flow.toggled.connect(label.setEnabled)
            if stage not in TEMPORAL_FLOW+EXPORT_FLOW:
                label=QLabel(STAGE_NAMES[stage]+note);label.setWordWrap(True);label.setToolTip(STAGE_NAMES[stage]+note);stages_layout.addWidget(label);check.toggled.connect(label.setEnabled)
        left.addWidget(group);left.addWidget(self.output_group);self.run_config=QWidget();config_layout=QVBoxLayout(self.run_config);presets=QHBoxLayout()
        self.all_button=QPushButton('完整流程');self.all_button.clicked.connect(lambda:self.select_stages(STAGES));render_button=QPushButton('仅生成输出');render_button.clicked.connect(lambda:self.select_stages(EXPORT_FLOW));presets.addWidget(self.all_button);presets.addWidget(render_button);config_layout.addLayout(presets)
        form=QFormLayout();self.workers=QSpinBox();self.workers.setRange(1,64);self.workers.setValue(6);self.frames=FrameHistoryBox();self.frames.setPlaceholderText('留空全部；前 10 帧填 0-9');self.workers.setToolTip('仅用于 CPU 逐帧可视化；不改变 SAM3、YOLO26x 或 WildDet3D 的 GPU 推理并行数。');self.frames.setToolTip('帧号统一从 0 开始：0 表示第一帧，0-9 表示前 10 帧。留空处理全部；指定帧从所选 KITTI / LMDB 读取；LMDB 数据保存到填写的导出目录，工作缓存单独存放。调试子集保留原始帧号。');form.addRow('可视化并行数',self.workers);config_layout.addLayout(form);frame_form=QFormLayout();frame_form.addRow('处理帧（从 0 开始）',self.frames);left.addLayout(frame_form);self.checks['render'].toggled.connect(self.update_worker_control)
        self.refresh_semantic=QCheckBox('重新生成语义输出（写入配置目录）');self.refresh_instances=QCheckBox('重新生成实例标签（写入配置目录）');config_layout.addWidget(self.refresh_semantic);self.refresh_classes=QComboBox();self.refresh_classes.addItem('强制重新生成：全部类别',None);
        for cid in range(10):self.refresh_classes.addItem(f'仅强制重新生成类别 {cid}',cid)
        self.refresh_classes.setToolTip('未勾选强制重新生成时，自动仅推理变化的提示词；勾选后可指定强制重算的类别。');config_layout.addWidget(self.refresh_classes);self.refresh_semantic.toggled.connect(self.update_refresh_controls);config_layout.addWidget(self.refresh_instances)
        self.run_button=QPushButton('开始执行');self.dry_button=QPushButton('预览执行命令');self.stop_button=QPushButton('停止');self.stop_button.setEnabled(False)
        self.run_button.clicked.connect(lambda:self.start(False));self.dry_button.clicked.connect(lambda:self.start(True));self.stop_button.clicked.connect(self.stop)
        config_layout.addWidget(self.dry_button);config_layout.addStretch()
        left.addWidget(QLabel('执行方式（单选）'))
        self.execution_modes=QButtonGroup(self)
        self.resume_cached=QRadioButton('复用缓存继续执行（默认）')
        self.recompute_postprocess=QRadioButton('复用 SAM，重算后处理')
        self.regenerate_all=QRadioButton('重新生成全部结果')
        for mode in (self.resume_cached,self.recompute_postprocess,self.regenerate_all):
            self.execution_modes.addButton(mode);left.addWidget(mode)
        self.resume_cached.setChecked(True)
        self.resume_cached.setToolTip('按所选阶段执行，输入和配置一致且缓存有效时跳过已完成处理。')
        self.recompute_postprocess.setToolTip('保留已有 SAM 和实例推理结果，强制重做位姿与关联、轨迹选点、位置优化、校验、可视化和标签导出。需先完成语义和实例标签。')
        self.regenerate_all.setToolTip('清理同一任务的旧结果和缓存，从模型推理开始重新生成，包括 SAM。')
        regenerate_note=QLabel('重算后处理保留已有 SAM 和实例结果；全部重新生成会清理缓存并重跑 SAM。')
        regenerate_note.setWordWrap(True);left.addWidget(regenerate_note)
        self.regenerate_all.toggled.connect(self.regeneration_changed)
        self.recompute_postprocess.toggled.connect(self.postprocess_changed)
        for b in [self.run_button,self.stop_button]:left.addWidget(b)
        b=QPushButton('打开输出目录');b.clicked.connect(self.open_result);left.addWidget(b)
        left.addStretch();self.results_panel=QWidget();right_layout=QVBoxLayout(self.results_panel);right_layout.setContentsMargins(0,0,0,0);tabs=QTabWidget();self.result_tabs=tabs;right_layout.addWidget(tabs,1);splitter.addWidget(self.results_panel);self.log=TaskLog();self.log.anchorClicked.connect(self.open_logged_visualization);tabs.addTab(self.log,'实时日志')
        preview=QWidget();pv=QVBoxLayout(preview);frame_row=QHBoxLayout();self.preview_frame=FrameSpinBox();self.preview_frame.setRange(0,999999);self.preview_frame.setToolTip('向上：上一帧；向下：下一帧。箭头、滚轮和键盘方向键一致，也可直接输入帧号。');self.preview_frame.setFixedWidth(90)
        self.previous_frame=QPushButton('◀ 上一帧');self.next_frame=QPushButton('下一帧 ▶');self.frame_slider=FrameSlider(Qt.Orientation.Horizontal);self.frame_slider.setTracking(False);self.frame_slider.setToolTip('拖动选择当前目录中已有的帧；松开后读取。');self.frame_count=QLabel()
        self.previous_frame.clicked.connect(self.preview_frame.stepUp);self.next_frame.clicked.connect(self.preview_frame.stepDown);self.frame_slider.valueChanged.connect(self.select_slider_frame)
        frame_row.addWidget(self.previous_frame);frame_row.addWidget(self.frame_slider,1);frame_row.addWidget(self.next_frame);frame_row.addWidget(QLabel('第'));frame_row.addWidget(self.preview_frame);frame_row.addWidget(self.frame_count);self.frame_controls=QWidget();self.frame_controls.setLayout(frame_row)
        mode_row=QHBoxLayout();mode_row.addWidget(self.projection_button);self.projection_button.setToolTip('点云深度投影与相机几何覆盖范围（虚线）；不表示完整标注或负样本区域');self.mask_button=QPushButton('查看语义 mask');self.mask_button.clicked.connect(lambda:self.show_label_mode('semantic'));mode_row.addWidget(self.mask_button);self.track_button=QPushButton('查看跟踪 ID');self.track_button.clicked.connect(lambda:self.show_label_mode('tracks'));mode_row.addWidget(self.track_button);self.combined_button=QPushButton('查看联合标注');self.combined_button.clicked.connect(lambda:self.show_label_mode('combined'));mode_row.addWidget(self.combined_button);self.map_button=QPushButton('查看全帧地图');self.map_button.clicked.connect(self.show_map_mode);mode_row.addWidget(self.map_button);pv.addLayout(mode_row)
        self.mode_buttons={'projection':self.projection_button,'semantic':self.mask_button,'tracks':self.track_button,'combined':self.combined_button,'map':self.map_button}
        self.mode_group=QButtonGroup(self);self.mode_group.setExclusive(True)
        for button in self.mode_buttons.values():
            button.setCheckable(True);self.mode_group.addButton(button)
            button.setStyleSheet('QPushButton:checked { background-color: #1769aa; color: white; border: 2px solid #0d4777; border-radius: 3px; font-weight: bold; }')
        self.combined_button.setChecked(True)
        self.track_controls=QWidget();track_row=QHBoxLayout(self.track_controls);track_row.setContentsMargins(0,0,0,0);track_row.addWidget(QLabel('目标 ID'));self.target_id=QComboBox();self.target_id.addItem('全部',None);track_row.addWidget(self.target_id);track_row.addWidget(QLabel('点云位置'));self.track_stage=QComboBox()
        for text,value in [('优化前','before'),('优化后','after'),('对比','compare')]:self.track_stage.addItem(text,value)
        self.track_stage.setCurrentIndex(self.track_stage.findData('after'))
        track_row.addWidget(self.track_stage);track_note=QLabel('虚线：点云范围；优化后按位置去离群');track_row.addWidget(track_note);track_row.addStretch();pv.addWidget(self.track_controls);self.track_controls.hide();self.track_data=None;self.track_focus_key=None
        self.target_id.currentIndexChanged.connect(self.update_track_view);self.track_stage.currentIndexChanged.connect(self.update_track_view)
        self.semantic_legend=QLabel();self.semantic_legend.setTextFormat(Qt.TextFormat.RichText);self.semantic_legend.setWordWrap(True);self.semantic_legend.hide();pv.addWidget(self.semantic_legend)
        self.frame_view_panel=QWidget();frame_layout=QVBoxLayout(self.frame_view_panel);frame_layout.setContentsMargins(0,0,0,0)
        self.camera_view=PointCanvas();self.bev_view=PointCanvas(bev=True);views=QSplitter();views.addWidget(self.camera_view);views.addWidget(self.bev_view);views.setSizes([700,500]);frame_layout.addWidget(views,1)
        self.camera_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus);self.bev_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.camera_view.target_selected.connect(self.select_target);self.bev_view.target_selected.connect(self.select_target)
        options=QHBoxLayout();options.addWidget(QLabel('点大小'));self.point_size=QSpinBox();self.point_size.setRange(1,8);self.point_size.setValue(3);self.point_size.valueChanged.connect(self.update_point_size);options.addWidget(self.point_size);reset=QPushButton('重置视图');reset.clicked.connect(lambda:(self.camera_view.reset_view(),self.bev_view.reset_view()));options.addWidget(reset);self.coverage_toggle=QPushButton('显示视觉范围');self.coverage_toggle.setCheckable(True);self.coverage_toggle.setChecked(True);self.coverage_toggle.setToolTip('显示或隐藏 BEV 的紫色范围虚线；不改变保存的 mask');self.coverage_toggle.toggled.connect(self.toggle_coverage);self.coverage_toggle.hide();options.addWidget(self.coverage_toggle);self.preview_status=QLabel('当前帧原始数据与最新标签；每秒检查更新，不读取历史可视化图。');self.preview_status.setWordWrap(True);options.addWidget(self.preview_status,1);frame_layout.addLayout(options)
        from .map_view import MapPage
        self.map_page=MapPage(self.map_context,self);self.preview_stack=QStackedWidget();self.preview_stack.addWidget(self.frame_view_panel);self.preview_stack.addWidget(self.map_page);pv.addWidget(self.preview_stack,1);pv.addWidget(self.frame_controls);tabs.addTab(preview,'结果可视化')
        bounds=QHBoxLayout();bounds.addWidget(QLabel('点云范围：XY ±'));self.xy_extent=QDoubleSpinBox();self.xy_extent.setRange(.5,200);self.xy_extent.setValue(20);self.xy_extent.setSuffix(' m');bounds.addWidget(self.xy_extent);bounds.addWidget(QLabel('高度 Z'));self.z_min=QDoubleSpinBox();self.z_min.setRange(-100,100);self.z_min.setValue(-3);self.z_min.setSuffix(' m');bounds.addWidget(self.z_min);bounds.addWidget(QLabel('至'));self.z_max=QDoubleSpinBox();self.z_max.setRange(-100,200);self.z_max.setValue(6);self.z_max.setSuffix(' m');bounds.addWidget(self.z_max);bounds.addStretch();config_layout.addLayout(bounds);range_note=QLabel('算法选点当前固定为 XY ±20 m、Z [-3,6) m；以上输入目前仅过滤查看，不会重新生成标签。');range_note.setWordWrap(True);config_layout.addWidget(range_note)
        for control in (self.xy_extent,self.z_min,self.z_max):control.valueChanged.connect(lambda:self.refresh_preview(True))
        self.preview_labels=True;self.preview_mode='combined';self.preview_key=None;self.frame_reader=None;self.preview_pending=False;self.preview_closing=False;self.preview_frame.valueChanged.connect(self.preview_frame_changed);tabs.currentChanged.connect(self.result_tab_changed);self.preview_timer=QTimer(self);self.preview_timer.setInterval(1000);self.preview_timer.timeout.connect(self.refresh_preview);self.preview_timer.start()
        for widget in [preview,tabs.tabBar(),*preview.findChildren(QWidget)]:widget.installEventFilter(self)
        self.help_page=QWidget();help_layout=QVBoxLayout(self.help_page);help_layout.addWidget(QLabel('当前数据与处理说明'));help_layout.addWidget(self.summary)
        self.format_help=QTextBrowser();self.format_help.setReadOnly(True)
        instructions=(REPO/'docs/kitti_input_format.md').read_text()
        minimal_example=sample_path('examples/kitti_minimal_two_frames')
        example_link=f'两帧最小示例目录（点击打开）：[{minimal_example}]({QUrl.fromLocalFile(str(minimal_example)).toString()})'
        instructions=instructions.replace('点云 `.bin` 是小端 float32',example_link+'\n\n点云 `.bin` 是小端 float32',1)
        self.format_help.setOpenLinks(False)
        self.format_help.anchorClicked.connect(lambda url:QDesktopServices.openUrl(url) if url.isLocalFile() else None)

        instructions+='\n\n## 处理与结果查看\n\n可视化并行数：仅控制 CPU 绘图任务，不改变 GPU 推理速度。\n\n处理帧：从 0 开始，0 表示第一帧，0-9 表示前 10 帧。留空处理全部；支持多个帧及包含两端的范围。指定帧时使用所选 KITTI / LMDB 原始数据运行所选流程。LMDB 导出目录直接保存标准数据；相同数据源、导出位置与帧范围复用独立工作缓存。\n\n语义叠加图：`任务目录/data/labels/zgc/overlay/`。\n\n联合审核图库：`任务目录/review/index.html`，需要执行逐帧联合可视化阶段。\n'
        self.format_help.setMarkdown(instructions)
        self.help_toc=QTreeWidget();self.help_toc.setHeaderLabel('目录');self.help_toc.setMinimumWidth(180)
        self.help_toc.setWordWrap(True)
        parents=[];block=self.format_help.document().begin()
        while block.isValid():
            level=block.blockFormat().headingLevel()
            if level:
                while parents and parents[-1][0]>=level:parents.pop()
                item=QTreeWidgetItem([block.text()]);item.setData(0,Qt.ItemDataRole.UserRole,block.position())
                if parents:parents[-1][1].addChild(item)
                else:self.help_toc.addTopLevelItem(item)
                parents.append((level,item))
            block=block.next()
        self.help_toc.expandAll();self.help_toc.itemClicked.connect(self.jump_to_help_section)
        self.help_splitter=QSplitter(Qt.Orientation.Horizontal)
        self.help_splitter.addWidget(self.help_toc);self.help_splitter.addWidget(self.format_help)
        self.help_splitter.setStretchFactor(0,0);self.help_splitter.setStretchFactor(1,1)
        self.help_splitter.setSizes([220,800]);help_layout.addWidget(self.help_splitter,1)
        self.open_format_example=QPushButton('打开 KITTI 示例目录');self.open_format_example.setEnabled(default_source('kitti').is_dir());self.open_format_example.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(default_source('kitti')))));help_layout.addWidget(self.open_format_example)
        tabs.addTab(self.run_config,'运行配置');tabs.addTab(self.help_page,'使用说明')
        self.visualization_panel=QWidget();visualization_layout=QVBoxLayout(self.visualization_panel);visualization_layout.setContentsMargins(0,0,0,0);result_row=QHBoxLayout();self.visualization_title=QLabel('当前阶段可视化');result_row.addWidget(self.visualization_title);self.visualization_path=QLineEdit();self.visualization_path.setReadOnly(True);result_row.addWidget(self.visualization_path,1);self.open_stage_button=QPushButton('打开目录');self.open_stage_button.clicked.connect(self.open_stage_visualization);result_row.addWidget(self.open_stage_button);self.copy_stage_button=QPushButton('复制路径');self.copy_stage_button.clicked.connect(lambda:QApplication.clipboard().setText(self.visualization_path.text()));result_row.addWidget(self.copy_stage_button);visualization_layout.addLayout(result_row);self.visualization_note=QLabel();self.visualization_note.setWordWrap(True);visualization_layout.addWidget(self.visualization_note);self.resource_status=QLabel('CPU -- · GPU -- · 显存 --');self.resource_status.setWordWrap(True);visualization_layout.addWidget(self.resource_status);from .resource_monitor import ResourceMonitor;self.resource_monitor=ResourceMonitor(self);self.resource_monitor.textChanged.connect(self.resource_status.setText);self.resource_monitor.start();right_layout.insertWidget(0,self.visualization_panel)
        for widget in (self.visualization_title,self.visualization_path,self.open_stage_button,self.copy_stage_button,self.visualization_note):widget.hide()
        self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);right_layout.addWidget(self.progress);self.frame_progress=QProgressBar();self.frame_progress.setRange(0,1);self.frame_progress.setValue(0);self.frame_progress.setFormat('当前阶段：等待开始');right_layout.addWidget(self.frame_progress);self.progress_text=QLabel('进度：等待开始');right_layout.addWidget(self.progress_text);self.state=QLabel('就绪；优化结果为待审核标签，轨迹补点单独标记。');right_layout.addWidget(self.state)
        self.stage_times=QLabel('阶段耗时：等待开始');self.stage_times.setWordWrap(False);self.stage_times.setFixedHeight(self.stage_times.fontMetrics().height()+4);self.stage_times.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed);self.stage_times.setTextFormat(Qt.TextFormat.PlainText);self.stage_times.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse);timing_row=QHBoxLayout();timing_row.addWidget(self.stage_times,1)
        self.stage_times_button=QPushButton('查看/复制');self.stage_times_button.clicked.connect(self.show_stage_times);timing_row.addWidget(self.stage_times_button);right_layout.addLayout(timing_row)
        splitter.setStretchFactor(0,0);splitter.setStretchFactor(1,1)
        self.editable=[self.source_group,group,self.output_group,self.all_button,render_button,self.workers,self.frames,self.refresh_semantic,self.refresh_instances,self.refresh_classes,self.dry_button,self.regenerate_all,self.resume_cached,self.recompute_postprocess];self.input_format.currentIndexChanged.connect(self.update_source_mode);self.update_source_mode();self.lmdb_output.editingFinished.connect(self.update_visualization_location);self.frames.editingFinished.connect(self.update_visualization_location);self.frames.textChanged.connect(self.update_frame_mode);self.frames.entry_selected.connect(self.select_test_history);self.frames.history_requested.connect(self.refresh_test_history);self.lmdb_root.editingFinished.connect(self.source_changed);self.refresh_test_history()

        self.ui_settings=QSettings(str(system_recipe_path().parent/'ui.ini'),QSettings.Format.IniFormat)
        modes={'cached':self.resume_cached,'postprocess':self.recompute_postprocess,'all':self.regenerate_all}
        modes.get(self.ui_settings.value('execution_mode','cached'),self.resume_cached).setChecked(True)
        for name,button in modes.items():
            button.clicked.connect(lambda checked,key=name:self.save_execution_mode(key))

    def save_execution_mode(self,mode):
        self.ui_settings.setValue('execution_mode',mode);self.ui_settings.sync()

    def load_saved_stage_times(self,context):
        if self.process.state()!=QProcess.ProcessState.NotRunning:return
        try:
            report=json.loads((Path(context['output'])/'run_report.json').read_text())
        except (OSError,ValueError):
            self.stage_times.setText('阶段耗时：暂无当前任务记录');self.stage_times.setToolTip('');return
        stages=[dict(entry) for entry in report.get('stages',[])]
        for entry in stages:
            if entry.get('status')=='running':entry['status']='未完成'
        details=stage_timing_text(stages,STAGE_NAMES)
        if details=='阶段耗时：等待开始':details='旧报告未记录分阶段耗时'
        total=report.get('elapsed_seconds')
        summary=f'最近一次处理耗时：{total:.1f} 秒' if total is not None else '最近一次处理：总耗时未记录'
        started=report.get('started_unix')
        date=datetime.fromtimestamp(started).strftime('%Y-%m-%d %H:%M:%S') if started is not None else '时间未记录'
        self.stage_times.setText(summary)
        self.stage_times.setToolTip(f'最近一次处理：{date}\n{summary}\n{details}')

    def show_stage_times(self):
        dialog=self.findChild(QDialog,'stage_timing_dialog')
        if dialog is None:
            dialog=QDialog(self);dialog.setObjectName('stage_timing_dialog');dialog.setWindowTitle('阶段耗时明细')
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose);dialog.resize(600,350)
            layout=QVBoxLayout(dialog)
            details=QPlainTextEdit(dialog);details.setObjectName('stage_timing_details');details.setReadOnly(True);layout.addWidget(details)
            buttons=QHBoxLayout();copy=QPushButton('复制全部',dialog);copy.setObjectName('copy_stage_times')
            copy.clicked.connect(lambda:QApplication.clipboard().setText(details.toPlainText()));buttons.addWidget(copy);buttons.addStretch()
            close=QPushButton('关闭',dialog);close.clicked.connect(dialog.close);buttons.addWidget(close);layout.addLayout(buttons)
        details=dialog.findChild(QPlainTextEdit,'stage_timing_details')
        details.setPlainText(self.stage_times.toolTip() or self.stage_times.text())
        dialog.show();dialog.raise_();dialog.activateWindow()

    def jump_to_help_section(self,item,column=0):
        cursor=self.format_help.textCursor();cursor.setPosition(item.data(0,Qt.ItemDataRole.UserRole))
        self.format_help.setTextCursor(cursor);self.format_help.ensureCursorVisible()

    def result_tab_changed(self,index):
        if index==1:
            if self.preview_mode!='map':self.camera_view.setFocus(Qt.FocusReason.OtherFocusReason)
            self.refresh_preview()

    def map_context(self):
        if self.process.state()!=QProcess.ProcessState.NotRunning:raise ValueError('请等待全部帧优化完成后再加载地图')
        return {**self.current_context(),'map_limits':self.preview_limits()}

    def show_map_mode(self):
        self.preview_mode='map';self.preview_labels=True;self.map_button.setChecked(True);self.coverage_toggle.hide()
        self.track_controls.hide();self.semantic_legend.hide();self.frame_controls.setEnabled(False);self.frame_controls.hide()
        self.preview_stack.setCurrentWidget(self.map_page);self.result_tabs.setCurrentIndex(1)
        try:
            if not self.map_page.matches_context(self.map_context()):self.map_page.refresh()
        except (OSError,ValueError,KeyError) as exc:self.map_page.status.setText(str(exc))

    def eventFilter(self,watched,event):
        if event.type()==QEvent.Type.KeyPress and self.result_tabs.currentIndex()==1 and self.preview_mode!='map' and event.modifiers()==Qt.KeyboardModifier.NoModifier:
            if not isinstance(watched,(QLineEdit,QAbstractSpinBox,QComboBox)):
                if event.key() in (Qt.Key.Key_Left,Qt.Key.Key_Up):
                    self.preview_frame.stepUp();return True
                if event.key() in (Qt.Key.Key_Right,Qt.Key.Key_Down):
                    self.preview_frame.stepDown();return True
        return super().eventFilter(watched,event)

    def refresh_test_history(self):
        source=(REPO/Path(self.lmdb_root.text()).expanduser()).resolve()
        entries=test_history(source)
        if self.input_format.currentData()=='kitti':entries=[e for e in entries if e['context'].get('input_format')=='kitti' and e['context'].get('camera')==self.camera_channel.currentData()]
        self.frames.set_history(entries)

    def select_test_history(self,entry):
        context=entry['context']
        self.result_contexts[self.result_context_key(self.base_context())]=context
        self.preview_key=None;self.track_data=None;self.track_focus_key=None
        self.update_visualization_location();self.refresh_preview(True)

    def record_test_start(self):
        if self.active_config and self.active_config.get('input_format') in ('lmdb','kitti') and not self.test_dry_run:
            record_test(REPO/Path(self.lmdb_root.text()).expanduser(),parse_frames(self.frames.text()),self.active_config)
            self.refresh_test_history()

    def update_visualization_location(self,stage=None):
        try:
            context=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            self.update_preview_frame_range(context)
            if stage is None and hasattr(self,'stage_times'):self.load_saved_stage_times(context)
            stage=stage or next((k for k,v in self.checks.items() if v.isChecked()),'render')
            path,note=visualization_location(context,stage)
            self.visualization_title.setText(STAGE_NAMES.get(stage,stage)+' · 可视化')
            self.visualization_path.setText(str(path) if path else '')
            self.copy_stage_button.setEnabled(path is not None)
            available=path is not None and path.is_dir()
            self.open_stage_button.setEnabled(available)
            images_ready=available and any(p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.svg','.html') for p in path.iterdir())
            self.visualization_note.setText(note+(' 目录尚未生成。' if path is not None and not available else ' 尚未生成可视化文件。' if available and not images_ready else ''))
        except (OSError,ValueError,KeyError) as exc:
            self.visualization_path.clear();self.visualization_note.setText(str(exc));self.open_stage_button.setEnabled(False);self.copy_stage_button.setEnabled(False)

    def open_stage_visualization(self):
        path=Path(self.visualization_path.text())
        if self.visualization_path.text() and path.is_dir():QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def update_preview_frame_range(self,context):
        out=Path(context['output'])
        index=out/'raw/index.json'
        if not index.is_file():index=Path(context.get('backend',{}).get('source_index',out/'data/labels/zgc/index.json'))
        if index.is_file():
            rows=json.loads(index.read_text())['rows']
            ids=[int(row.get('original_frame_id',row.get('source_frame_id',Path(row['path']).stem))) for row in rows]
        else:
            ids=parse_frames(self.frames.text())
            if ids is None and self.input_format.currentData() in ('lmdb','kitti'):
                from .raw_input import frame_records
                try:ids=[int(r['global_frame_id']) for r in frame_records(REPO/Path(self.lmdb_root.text()).expanduser(),camera=self.camera_channel.currentData() if self.input_format.currentData()=='kitti' else None)[1]]
                except (OSError,ValueError):ids=[]
        ids=sorted(set(ids)) if ids else []
        self.preview_frame.frame_ids=ids
        blocked=self.preview_frame.blockSignals(True)
        self.preview_frame.setRange(ids[0] if ids else 0,ids[-1] if ids else 999999)
        if ids:self.preview_frame.setValue(min(ids,key=lambda frame:abs(frame-self.preview_frame.value())))
        self.preview_frame.blockSignals(blocked)
        self.sync_frame_navigation()

    def sync_frame_navigation(self):
        ids=getattr(self.preview_frame,'frame_ids',[])
        value=self.preview_frame.value()
        position=bisect.bisect_left(ids,value) if ids else value
        blocked=self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0,len(ids)-1 if ids else self.preview_frame.maximum())
        if not self.frame_slider.isSliderDown():self.frame_slider.setValue(position)
        self.frame_slider.blockSignals(blocked)
        self.previous_frame.setEnabled(value>self.preview_frame.minimum())
        self.next_frame.setEnabled(value<self.preview_frame.maximum())
        self.frame_count.setText(f'帧 / 共 {len(ids)} 帧' if ids else '帧 / 总数待读取')
        self.frame_count.setToolTip(f'原始帧范围：{ids[0]}–{ids[-1]}；帧号从 0 开始，缺失帧自动跳过。' if ids else '帧号从 0 开始；读取目录索引后显示实际帧数。')

    def select_slider_frame(self,position):
        ids=getattr(self.preview_frame,'frame_ids',[])
        self.preview_frame.setValue(ids[position] if ids else position)

    def preview_frame_changed(self):
        ids=getattr(self.preview_frame,'frame_ids',[])
        value=self.preview_frame.value()
        if ids and value not in ids:
            self.preview_frame.setValue(min(ids,key=lambda frame:abs(frame-value)))
            return
        self.sync_frame_navigation()
        self.refresh_preview(True)

    def current_context(self):
        context=self.base_context()
        source=(REPO/Path(self.lmdb_root.text()).expanduser()).resolve()
        frames=parse_frames(self.frames.text())
        context=self.result_contexts.get(self.result_context_key(context),latest_result(context,source,frames))
        return context

    def result_context_key(self,context):
        return (context['output'],str((REPO/Path(self.lmdb_root.text()).expanduser()).resolve()),tuple(parse_frames(self.frames.text()) or []))

    def base_context(self):
        frames=parse_frames(self.frames.text());kind=self.input_format.currentData()
        camera=self.camera_channel.currentData() if kind=='kitti' else None
        if kind=='lmdb':
            dataset=(REPO/Path(self.lmdb_output.text()).expanduser()).resolve()
            source=(REPO/Path(self.lmdb_root.text()).expanduser()).resolve()
            out=work_directory(source,dataset,frames)
            return dict(output=str(out),review_output=str(out/'review'),input_format='lmdb',dataset_output=str(dataset))
        if frames is not None:
            source=Path(self.lmdb_root.text()).expanduser();source=source if source.is_absolute() else REPO/source
            out=temporary_debug_output(source,frames)
            if camera:out=out/camera
            return dict(output=str(out),review_output=str(out/'review'),input_format='kitti' if kind=='kitti' else 'lmdb',temporary_debug=True,**({'camera':camera} if camera else {}))
        if kind in ('lmdb','kitti'):
            out=Path(self.lmdb_output.text()).expanduser();out=out if out.is_absolute() else REPO/out
            if camera:out=out/camera
            return dict(output=str(out),review_output=str(out/'review'),input_format=kind,**({'camera':camera} if camera else {}))
        return load_config(REPO/self.profile.text())

    def source_changed(self):
        if self.input_format.currentData()=='kitti' and self.lmdb_root.text().strip():
            source=(REPO/Path(self.lmdb_root.text()).expanduser()).resolve()
            self.lmdb_root.setText(os.path.relpath(source,REPO));self.lmdb_root.setCursorPosition(0)
        self.refresh_camera_channels();self.refresh_test_history();self.update_visualization_location()

    def refresh_camera_channels(self):
        if self.input_format.currentData()!='kitti':return
        from .kitti_input import camera_channels
        previous=self.camera_channel.currentData();blocked=self.camera_channel.blockSignals(True)
        self.camera_channel.clear()
        try:
            channels=camera_channels(REPO/Path(self.lmdb_root.text()).expanduser())
            for channel in channels:self.camera_channel.addItem(channel,channel)
            chosen=previous if previous in channels else 'image_2' if 'image_2' in channels else channels[0] if channels else None
            if chosen:self.camera_channel.setCurrentIndex(self.camera_channel.findData(chosen))
        except (OSError,ValueError):pass
        finally:self.camera_channel.blockSignals(blocked)

    def camera_changed(self):
        self.preview_key=None;self.track_data=None;self.track_focus_key=None
        self.refresh_test_history();self.update_visualization_location()

    def update_source_mode(self):
        raw=self.input_format.currentData() in ('lmdb','kitti')
        for widget in [self.lmdb_root,self.lmdb_browse,self.lmdb_output]:widget.setEnabled(raw)
        mode='raw' if raw else 'configured'
        changed=mode!=self._profile_mode
        if changed:
            if raw:self.cached_profile=self.profile.text();self.profile.setText(self.recipe_profile)
            else:self.recipe_profile=self.profile.text();self.profile.setText(self.cached_profile)
            self._profile_mode=mode
        kind=self.input_format.currentData()
        source_changed=kind!=self._source_mode
        if source_changed:
            if self._source_mode in ('lmdb','kitti'):
                self.source_paths[self._source_mode]=self.lmdb_root.text();self.output_paths[self._source_mode]=self.lmdb_output.text()
            if kind in ('lmdb','kitti'):
                self.lmdb_root.setText(self.source_paths[kind]);self.lmdb_output.setText(self.output_paths[kind])
            self._source_mode=kind;self.refresh_camera_channels();self.refresh_test_history()
        self.source_label.setText('KITTI 序列目录' if kind=='kitti' else 'LMDB 采集根目录')
        self.output_label.setText('处理结果目录' if kind=='kitti' else 'LMDB 导出目录')
        self.lmdb_root.setPlaceholderText('包含 image_*、velodyne 和标定的目录' if kind=='kitti' else '包含 index.lmdb 和 shards 的目录')
        self.camera_label.setVisible(kind=='kitti');self.camera_channel.setVisible(kind=='kitti')
        self.update_frame_mode()
        if source_changed and raw:self.preview_frame.setValue(self.preview_frame.minimum())
        self.profile_label.setText('标注配置（类别 / 提示词）' if raw else '处理配置（已有缓存）')
        for widget in (self.profile_label,self.profile,self.profile_browse):widget.setVisible(not raw)
        self.profile.setReadOnly(raw)
        self.profile.setEnabled(True);self.profile_browse.setEnabled(not raw);self.recipe_button.setEnabled(True);self.all_button.setEnabled(True);self.refresh_semantic.setEnabled(True);self.refresh_instances.setEnabled(True);self.update_refresh_controls()
        self.update_worker_control()
        for check in self.checks.values():check.setEnabled(True)
        self.checks['render'].setText(STAGE_NAMES['render'])
        if raw:self.select_stages(STAGES)
        self.update_summary()

    def update_frame_mode(self):
        subset=bool(self.frames.text().strip()) or self.regenerate_all.isChecked();raw=self.input_format.currentData() in ('lmdb','kitti')
        self.lmdb_root.setEnabled(raw or subset);self.lmdb_browse.setEnabled(raw or subset)
        self.update_refresh_controls();self.update_visualization_location()

    def regeneration_changed(self,checked):
        if checked:self.select_stages(STAGES)
        self.update_frame_mode()

    def postprocess_changed(self,checked):
        if checked:
            self.refresh_semantic.setChecked(False);self.refresh_instances.setChecked(False)
            self.select_stages(TEMPORAL_FLOW+EXPORT_FLOW)
        self.update_refresh_controls()

    def update_refresh_controls(self):
        full=self.regenerate_all.isChecked() or self.recompute_postprocess.isChecked()
        self.refresh_semantic.setEnabled(not full);self.refresh_instances.setEnabled(not full)
        self.refresh_classes.setEnabled(not full and (self.input_format.currentData() in ('lmdb','kitti') or bool(self.frames.text().strip())) and self.refresh_semantic.isChecked())

    def debug_prompts(self):
        from .prompt_debug_dialog import PromptDebugDialog
        from .live_preview import frame_inputs
        from PIL import Image
        try:
            context=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            if not self.preview_labels and hasattr(self,'projection_context') and self.projection_context['output']==context['output']:context=self.projection_context
            frame=self.camera_view.data['frame'] if self.camera_view.data is not None else self.preview_frame.value()
            saved=Path(context['output'])/'recipe.json'
            recipe=load_recipe(REPO/self.profile.text() if self.input_format.currentData() in ('lmdb','kitti') else system_recipe_path() if system_recipe_path().is_file() else saved if saved.is_file() else self.recipe_profile)
            image=None;image_path=None
            try:
                path=frame_inputs(context,frame,False)['image']
                if path.is_file():image=Image.open(path).convert('RGB');image_path=path
            except (OSError,ValueError,KeyError):pass
            if image is None and self.input_format.currentData() in ('lmdb','kitti'):
                from .raw_input import source_image
                image=source_image(REPO/Path(self.lmdb_root.text()).expanduser(),frame,camera=self.camera_channel.currentData() if self.input_format.currentData()=='kitti' else None)
            dialog=PromptDebugDialog(recipe,image,self,image_path=image_path)
            def use_recipe(path):
                self.recipe_profile=path
                if self.input_format.currentData() in ('lmdb','kitti'):self.profile.setText(path);self.update_summary()
            dialog.recipe_saved.connect(use_recipe)
            dialog.exec()
        except (OSError,ValueError,KeyError) as exc:self.state.setText(f'无法打开提示词调试：{exc}')

    def browse_lmdb(self):
        path=QFileDialog.getExistingDirectory(self,'选择 KITTI 序列目录' if self.input_format.currentData()=='kitti' else '选择 LMDB 采集根目录',str((REPO/Path(self.lmdb_root.text()).expanduser()).resolve()) if self.lmdb_root.text() else str(Path.home()))
        if path:self.lmdb_root.setText(path);self.source_changed()

    def update_summary(self):
        try:
            if self.input_format.currentData() in ('lmdb','kitti'):
                load_recipe(REPO/self.profile.text());self.recipe_profile=self.profile.text()
                if self.input_format.currentData()=='kitti':
                    self.summary.setText('读取所选 image_* 相机、velodyne 点云、相机标定、时间戳与位姿。每次处理一个相机通道，结果按通道分开保存。');return
                self.summary.setText('LMDB 输入：先只读解析 index.lmdb 与 shards，建立内部缓存，再导出 KITTI 风格数据到填写的导出目录。\n导出成功后才执行所选标注阶段；后续算法读取内部缓存。SAM 等模型结果按执行方式复用或重算。\n标签保存到导出目录的 labels/；工作缓存、日志与审核图库单独保存。');return
            c=load_config(REPO/self.profile.text());self.summary.setText(f"数据适配器：{c['adapter']}\n输出：{c['output']}\n语义、视觉 mask、检测记录、跟踪 ID 和点云标签均保留。")
            self.update_visualization_location()
            if c.get('diagnostic_only'):
                self.select_stages(['render','catalog']);self.summary.setText(self.summary.text()+'\n当前为随包测试集：仅支持可视化和联合索引。')
                index=Path(c['backend']['source_index']);index=index if index.is_absolute() else REPO/index
                rows=json.loads(index.read_text())['rows']
                if rows:self.preview_frame.setValue(int(Path(rows[0]['path']).stem))
        except Exception as exc:self.summary.setText(f'配置错误：{exc}')

    def browse_profile(self):
        if self.input_format.currentData() in ('lmdb','kitti'):return
        path,_=QFileDialog.getOpenFileName(self,'选择标注配置',str((REPO/self.profile.text()).parent),'JSON (*.json)')
        if path:self.profile.setText(os.path.relpath(path,REPO));self.update_summary()

    def update_worker_control(self):
        self.workers.setEnabled(self.checks['render'].isChecked() and self.process.state()==QProcess.ProcessState.NotRunning)

    def toggle_temporal_flow(self,checked):
        selected=[k for k,v in self.checks.items() if k not in TEMPORAL_FLOW and v.isChecked()]
        self.select_stages(selected+list(TEMPORAL_FLOW) if checked else selected)

    def toggle_export_flow(self,checked):
        selected=[k for k,v in self.checks.items() if k not in EXPORT_FLOW and v.isChecked()]
        self.select_stages(selected+list(EXPORT_FLOW) if checked else selected)

    def select_stages(self,stages):
        stages=set(stages)
        if stages.intersection(TEMPORAL_FLOW):stages.update(TEMPORAL_FLOW)
        if stages.intersection(EXPORT_FLOW):stages.update(EXPORT_FLOW)
        self.export_flow.blockSignals(True);self.export_flow.setChecked(set(EXPORT_FLOW).issubset(stages));self.export_flow.blockSignals(False)
        self.temporal_flow.blockSignals(True);self.temporal_flow.setChecked(set(TEMPORAL_FLOW).issubset(stages));self.temporal_flow.blockSignals(False)
        for key,check in self.checks.items():check.setChecked(key in stages)
        self.update_worker_control()
        self.update_visualization_location()

    def start(self,dry_run=False):
        if self.process.state()!=QProcess.ProcessState.NotRunning:return
        if self.map_page.reader is not None:self.state.setText('正在读取地图，请等待完成后再启动处理。');return
        try:
            c=self.current_context();selected=[k for k,v in self.checks.items() if v.isChecked()];regenerate=self.regenerate_all.isChecked()
            recipe=self.profile.text() if self.input_format.currentData() in ('lmdb','kitti') else str(system_recipe_path()) if system_recipe_path().is_file() else str(Path(c['output'])/'recipe.json') if (Path(c['output'])/'recipe.json').is_file() else self.recipe_profile
            postprocess=self.recompute_postprocess.isChecked()
            if postprocess:
                if c.get('input_format') not in ('lmdb','kitti'):raise ValueError('重算后处理需要选择 LMDB 或 KITTI 原始序列任务')
                selected=list(TEMPORAL_FLOW+EXPORT_FLOW);self.select_stages(selected)
            if regenerate:
                selected=list(STAGES);self.select_stages(selected);previous=c
                source=REPO/Path(self.lmdb_root.text()).expanduser();frames=parse_frames(self.frames.text())
                old_results=clear_previous_results(self.base_context(),previous,source,frames,dry_run=True)
                recipe_path=(REPO/Path(recipe)).resolve();snapshot=load_recipe(recipe_path)
                base=Path(self.base_context()['output']).resolve();stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
                out=base.parent/(base.name+'_runs')/stamp
                c={**c,'output':str(out),'review_output':str(out/'review'),'input_format':c.get('input_format','lmdb'),'temporary_debug':c.get('temporary_debug',False)}
                copy_recipe=not dry_run and any(recipe_path.is_relative_to(path) for path in old_results)
                if copy_recipe:recipe=str(out/'recipe.json')
            if c.get('input_format') in ('lmdb','kitti'):
                args=annotate_command(self.lmdb_root.text(),c['output'],selected,self.workers.value(),self.frames.text(),dry_run,regenerate or self.refresh_semantic.isChecked(),regenerate or self.refresh_instances.isChecked(),recipe_path=recipe,semantic_classes=[self.refresh_classes.currentData()] if not regenerate and self.refresh_semantic.isChecked() and self.refresh_classes.currentData() is not None else None,input_format=c.get('input_format'),camera=c.get('camera'),dataset_output=c.get('dataset_output'),refresh_postprocess=postprocess)
            else:args=command_for(self.profile.text(),selected,self.workers.value(),self.frames.text(),dry_run,self.refresh_semantic.isChecked(),self.refresh_instances.isChecked())
            if regenerate and not dry_run:
                clear_previous_results(self.base_context(),previous,source,frames)
                if copy_recipe:
                    out.mkdir(parents=True,exist_ok=True);Path(recipe).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
                self.result_contexts={key:value for key,value in self.result_contexts.items() if Path(value['output']).resolve() not in old_results}
                remember_result(self.base_context(),source,frames,c);self.refresh_test_history()
        except Exception as exc:self.state.setText(f'无法启动：{exc}');return
        selected_frames=parse_frames(self.frames.text())
        if selected_frames and self.preview_frame.value() not in selected_frames:self.preview_frame.setValue(selected_frames[0])
        if regenerate and not dry_run:
            self.result_contexts[self.result_context_key(self.base_context())]=c
            self.preview_key=None;self.track_data=None;self.track_focus_key=None;self.semantic_legend.hide()
            for view in (self.camera_view,self.bev_view):view.data=None;view.groups=[];view.reset_view()
        self.stage_times.setText('阶段耗时：等待开始');self.stage_times.setToolTip('')
        self.active_config=c;self.selected_stages=selected;self.stopping=False;self.test_dry_run=dry_run
        self.offsets={Path(c['output'])/'logs'/f'{stage}.log':(Path(c['output'])/'logs'/f'{stage}.log').stat().st_size for stage in selected if (Path(c['output'])/'logs'/f'{stage}.log').exists()}
        self.log.clear();self.log.append_directory(Path(c['review_output'])/'visualization');self.log.appendPlainText('执行：'+shlex.join([sys.executable,*args]));self.progress.setRange(0,0);self.frame_progress.setRange(0,0);self.frame_progress.setFormat('当前阶段：准备中');self.progress_text.setText('进度：准备开始');self.state.setText('正在预览…' if dry_run else '正在执行…')
        if c.get('dataset_output'):self.log.append_directory(c['dataset_output'],label='标准数据目录')
        if regenerate:self.log.appendPlainText('预览：将清理同一任务的旧结果并重新生成。' if dry_run else f'重新生成全部结果：已清理 {len(old_results)} 个旧任务目录，不复用处理缓存。');self.log.append_directory(c['output'])
        if c.get('temporary_debug'):
            self.log.appendPlainText('临时调试目录：'+c['output']);self.log.append_directory(c['output']);self.log.appendPlainText('仅使用所选帧；正式输出目录保持不变。系统可能清理临时文件，请及时保存需要的结果。')
        for widget in self.editable:widget.setEnabled(False)
        self.run_button.setEnabled(False);self.dry_button.setEnabled(False);self.stop_button.setEnabled(True)
        self.process.setWorkingDirectory(str(REPO));self.process.start(sys.executable,args);self.timer.start()

    def read_stdout(self):self.log.append_stream(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))

    def open_logged_visualization(self,url):
        if not url.isLocalFile():return
        path=Path(url.toLocalFile())
        if path.is_dir():QDesktopServices.openUrl(url)
        else:self.state.setText(f'可视化目录尚未生成：{path}')

    def read_stage_log(self):
        if not self.active_config:return
        out=Path(self.active_config['output'])
        report=out/'run_report.json'
        try:
            data=json.loads(report.read_text());completed=sum(s['status']=='complete' for s in data['stages'])
            self.progress.setRange(0,max(1,len(self.selected_stages)));self.progress.setValue(completed);self.progress.setFormat('处理步骤 %v/%m')
            timing_details=stage_timing_text(data['stages'],STAGE_NAMES)
            self.stage_times.setText('阶段耗时：'+timing_details.splitlines()[-1] if '\n' in timing_details else timing_details)
            self.stage_times.setToolTip(timing_details)
            current=data['stages'][-1]['stage'] if data['stages'] else ''
            status_text=f"进度：已完成 {completed}/{len(self.selected_stages)} 步 · 当前阶段：{STAGE_NAMES.get(current,current)}"
            self.update_visualization_location(current)
            path,_=visualization_location(self.active_config,current)
            if path is not None:self.log.append_directory(path)
            self.state.setText(('正在停止…' if self.stopping else '运行中：')+STAGE_NAMES.get(current,current))
            self.frame_progress.setRange(0,0);self.frame_progress.setFormat('当前阶段：初始化或处理中')
            if data['status']=='complete':
                self.frame_progress.setRange(0,1);self.frame_progress.setValue(1);self.frame_progress.setFormat('当前阶段：完成')
            else:
                try:
                    item=json.loads((out/'progress.json').read_text())
                    if item['stage']==current and item['total']>0:
                        self.frame_progress.setRange(0,item['total']);self.frame_progress.setValue(item['completed'])
                        self.frame_progress.setFormat(f"{STAGE_NAMES.get(current,current)}：%v/%m（%p%）")
                        status_text+=f" · {item['completed']}/{item['total']}（{100*item['completed']/item['total']:.1f}%）"
                        suffix=f" · {item['detail']}" if item.get('detail') else ''
                        if item.get('frame') is not None:suffix+=f" · 帧 {item['frame']}"
                        status_text+=suffix
                        self.state.setText(self.state.text()+suffix)
                except (OSError,ValueError,KeyError):pass
            self.progress_text.setText(status_text)
        except (OSError,ValueError,KeyError):
            raw_status=out/'raw'/'status.json'
            try:
                item=json.loads(raw_status.read_text());done=item.get('completed',item.get('imported_frames',0));total=item.get('total',item.get('total_recording_frames',0))
                if item.get('status')=='complete':done=total=item.get('imported_frames',done)
                phase='转换 KITTI' if item.get('phase')=='kitti' else '读取 LMDB'
                state=item.get('status','running')
                self.progress.setRange(0,max(1,total));self.progress.setValue(min(done,total) if total else 0);self.progress.setFormat('原始导入 %v/%m' if total else '原始导入')
                count_text=f'{done}/{total}' if total else f'已导入 {done}'
                self.progress_text.setText(f'进度：{phase} {count_text} 帧 · {state}')
                self.frame_progress.setRange(0,0);self.frame_progress.setFormat('当前阶段：'+phase)
                self.state.setText(f'正在执行：{phase}（{count_text} 帧）' if state=='running' else f'原始导入{("完成" if state=="complete" else "失败或已停止")}：{count_text} 帧')
            except (OSError,ValueError,KeyError):
                self.progress_text.setText('进度：正在初始化')
        for stage in self.selected_stages:
            path=out/'logs'/f'{stage}.log'
            if not path.exists():continue
            start=self.offsets.get(path,0)
            with path.open('rb') as stream:stream.seek(start);text=stream.read();self.offsets[path]=stream.tell()
            if text:
                lines=[line for line in text.decode('utf-8',errors='replace').splitlines() if not line.startswith(('fresh_semantics ','fresh inference '))]
                if lines:self.log.appendPlainText('\n'.join(lines))

    def stop(self):
        self.stopping=True;self.stop_button.setEnabled(False);self.process.terminate();self.state.setText('正在停止处理进程…')

    def finished(self,code,status):
        self.stopping=self.stopping or code==130
        if self.active_config and self.active_config.get('input_format') in ('lmdb','kitti') and not getattr(self,'test_dry_run',False):
            finish_test(self.active_config,'cancelled' if self.stopping else 'complete' if code==0 else 'failed');self.refresh_test_history()
        self.read_stdout();self.log.append_stream('',final=True);self.read_stage_log();self.timer.stop();self.progress.setRange(0,1);self.progress.setValue(1 if code==0 else 0);self.progress.setFormat('%p%');self.frame_progress.setRange(0,max(1,self.frame_progress.maximum()));self.frame_progress.setFormat('当前阶段：完成' if code==0 else '当前阶段：已停止' if self.stopping else '当前阶段：失败')
        for widget in self.editable:widget.setEnabled(True)
        self.run_button.setEnabled(True);self.dry_button.setEnabled(True);self.stop_button.setEnabled(False)
        selected=[k for k,v in self.checks.items() if v.isChecked()];self.update_source_mode();self.select_stages(selected)
        if self.active_config and self.active_config.get('input_format') in ('lmdb','kitti') and code==0:
            ids=parse_frames(self.frames.text()) or []
            if not ids:
                try:ids=json.loads((Path(self.active_config['output'])/'report.json').read_text()).get('frame_ids',[])
                except (OSError,ValueError):ids=[]
            self.preview_frame.setValue(int(ids[0]) if ids else 0)
        self.state.setText('已停止，已完成的输出保留。' if self.stopping else '处理完成。' if code==0 else f'处理失败（退出码 {code}），请查看日志。')
        self.progress_text.setText('进度：已停止' if self.stopping else f'进度：处理完成 {len(self.selected_stages)}/{len(self.selected_stages)} 步' if code==0 else f'进度：处理失败（退出码 {code}）')
        if self.active_config:
            try:
                report=json.loads((Path(self.active_config['output'])/'run_report.json').read_text())
                if report.get('stages'):self.update_visualization_location(report['stages'][-1]['stage'])
            except (OSError,ValueError,KeyError):pass
        if code==0 and not getattr(self,'test_dry_run',False) and any(s in self.selected_stages for s in ('optimize','render')):
            from .optimized_map import optimized_map_ready
            try:
                if optimized_map_ready(self.map_context()):
                    self.map_page.context_key=None;self.map_page.refresh()
            except (OSError,ValueError,KeyError) as exc:self.map_page.status.setText(str(exc))
        if self.close_pending:self.close()

    def process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:self.finished(-1,QProcess.ExitStatus.CrashExit)

    def open_result(self):
        try:
            c=self.current_context();path=Path(c['output'])
            if c.get('input_format')=='lmdb':
                portable=Path(c['dataset_output']) if c.get('dataset_output') else path/'kitti'
                if not (portable/'metadata.npz').is_file():
                    progress=path/'raw/status.json';detail='正在准备转换数据。'
                    if progress.is_file():
                        status=json.loads(progress.read_text());done=status.get('completed',0);total=status.get('total',0)
                        if status.get('kitti_export_status')=='failed':
                            self.state.setText('KITTI 导出失败：'+status.get('kitti_export_error','请查看日志'));return
                        if status.get('status') in ('cancelled','failed'):
                            self.state.setText('数据导出已停止或失败：'+status.get('error','请查看日志'));return
                        if status.get('phase')=='kitti':detail=f'正在转换 KITTI：{done}/{total} 帧。'
                        else:detail=f'正在读取 LMDB：{done}/{total} 帧。' if total and done<total else '正在整理 KITTI 数据。'
                    self.state.setText('标准数据目录尚未生成；'+detail+'转换完成后可打开。');return
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(portable)));return
            report=path/'label_export.json'
            if report.is_file():
                labels=Path(json.loads(report.read_text())['output'])
                if labels.is_dir():path=labels
            if not path.exists():raise FileNotFoundError(path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:self.state.setText(str(exc))

    def projection_frame_context(self,context,frame):
        from .raw_input import read_raw
        out=Path(context['output']);index=out/'raw/index.json'
        if index.is_file() or Path(context.get('backend',{}).get('source_index',out/'data/labels/zgc/index.json')).is_file():return context
        source=(REPO/Path(self.lmdb_root.text()).expanduser()).resolve();camera=context.get('camera')
        # Single-frame preview caches belong to a source, camera and requested frame.
        cache=out/'projection_raw'/(camera or 'lmdb')/temporary_debug_output(source,[frame]).name
        if not (cache/'index.json').is_file():read_raw(source,cache,frames=[frame],camera=camera,kitti_output=False)
        return {**context,'backend':{'source_index':str(cache/'index.json')}}

    def show_projection(self):
        try:
            c=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            frame=self.preview_frame.value();selected=parse_frames(self.frames.text())
            if selected and frame not in selected:frame=selected[0];self.preview_frame.setValue(frame)
            self.projection_context=self.projection_frame_context(c,frame)
            if self.preview_mode=='tracks':
                self.track_focus_key=None;self.camera_view.reset_view();self.bev_view.reset_view()
            self.preview_stack.setCurrentWidget(self.frame_view_panel);self.frame_controls.setEnabled(True);self.frame_controls.show()
            self.track_controls.hide();self.preview_labels=False;self.preview_mode='projection';self.projection_button.setChecked(True);self.coverage_toggle.show();self.result_tabs.setCurrentIndex(1);self.refresh_preview(True)
        except Exception as exc:
            self.mode_buttons[self.preview_mode].setChecked(True);self.preview_status.setText(f'无法查看投影：{exc}')

    def show_label_mode(self,mode):
        self.preview_stack.setCurrentWidget(self.frame_view_panel);self.frame_controls.setEnabled(True);self.frame_controls.show()
        if self.preview_mode=='tracks' and mode!='tracks':
            self.track_focus_key=None;self.camera_view.reset_view();self.bev_view.reset_view()
        self.coverage_toggle.hide();self.track_controls.setVisible(mode=='tracks');self.preview_labels=True;self.preview_mode=mode;self.mode_buttons[mode].setChecked(True);self.result_tabs.setCurrentIndex(1);self.refresh_preview(True)

    def load_image(self):self.refresh_preview(True)

    def toggle_coverage(self,checked):
        self.bev_view.show_coverage=checked;self.bev_view.update()

    def update_point_size(self,value):
        for view in (self.camera_view,self.bev_view):view.point_size=value;view.update()

    def preview_limits(self):return (self.xy_extent.value(),self.z_min.value(),self.z_max.value())

    def refresh_preview(self,force=False):
        if not hasattr(self,'preview_labels') or not hasattr(self,'result_tabs'):return
        if not force and self.result_tabs.currentIndex()!=1:return
        if self.preview_mode=='map':
            try:
                if not self.map_page.matches_context(self.map_context()):self.map_page.refresh()
            except (OSError,ValueError,KeyError) as exc:self.map_page.status.setText(str(exc))
            return
        if self.frame_reader is not None:
            if force:self.preview_pending=True
            return
        try:
            c=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            self.update_preview_frame_range(c)
            frame=self.preview_frame.value()
            if not self.preview_labels:
                c=self.projection_frame_context(c,frame);self.projection_context=c
            key=(c['output'],frame,self.preview_labels,self.preview_limits(),self.preview_mode,fingerprint(c,frame,self.preview_labels))
            if not force and key==self.preview_key:return
            reader=FrameReader(c,frame,self.preview_labels,key,self,limits=self.preview_limits(),mode=self.preview_mode);self.frame_reader=reader
            reader.loaded.connect(self.preview_loaded);reader.failed.connect(lambda message,k=key:self.preview_failed(k,message));reader.finished.connect(self.preview_reader_finished);reader.start()
        except (OSError,ValueError,KeyError,StopIteration) as exc:self.preview_status.setText('等待当前帧数据：'+str(exc))

    def preview_failed(self,key,message):
        if key[1]==self.preview_frame.value() and key[2]==self.preview_labels and key[3]==self.preview_limits() and key[4]==self.preview_mode:self.preview_status.setText('等待当前帧数据：'+message)

    def update_semantic_legend(self,classes):
        self.semantic_legend.setText(semantic_legend_html(classes))
        self.semantic_legend.setVisible(bool(classes))

    def preview_loaded(self,key,data):
        try:
            context=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            if key[0]!=context['output']:return
        except (OSError,ValueError,KeyError):return
        if key[1]!=self.preview_frame.value() or key[2]!=self.preview_labels or key[3]!=self.preview_limits() or key[4]!=self.preview_mode:return
        self.preview_key=key
        if data['mode']=='tracks':
            self.track_data=data
            selected=self.target_id.currentData();ids=sorted({t['id'] for t in data['targets']})
            blocked=self.target_id.blockSignals(True);self.target_id.clear();self.target_id.addItem('全部',None)
            for tid in ids:self.target_id.addItem(str(tid),tid)
            if selected is not None and selected not in ids:self.target_id.addItem(f'{selected}（当前帧无此 ID）',selected)
            self.target_id.setCurrentIndex(max(0,self.target_id.findData(selected)));self.target_id.blockSignals(blocked)
            self.update_track_view()
        else:
            self.camera_view.set_frame(data);self.bev_view.set_frame(data);self.preview_status.setText(data['status'])
        self.update_semantic_legend(data.get('semantic_classes',[]))

    def select_target(self,tid):
        index=self.target_id.findData(tid)
        if index>=0:self.target_id.setCurrentIndex(index)

    def update_track_view(self):
        if self.preview_mode!='tracks' or self.track_data is None:return
        # Never reuse a previous frame while a new asynchronous read is pending.
        if self.track_data['frame']!=self.preview_frame.value():return
        try:
            context=self.active_config if self.process.state()!=QProcess.ProcessState.NotRunning and self.active_config else self.current_context()
            if self.preview_key[0]!=context['output']:return
        except (OSError,ValueError,KeyError):return
        selected=self.target_id.currentData();data=track_view(self.track_data,selected,self.track_stage.currentData())
        self.camera_view.set_frame(data);self.bev_view.set_frame(data);self.preview_status.setText(data['status'])
        focus=(self.preview_key[0],data['frame'],selected)
        if focus!=self.track_focus_key:
            self.track_focus_key=focus
            for view,name in ((self.camera_view,'focus_camera'),(self.bev_view,'focus_bev')):
                if selected is None:view.reset_view()
                else:view.fit_target(data[name])

    def preview_reader_finished(self):
        reader=self.frame_reader;self.frame_reader=None
        if reader is not None:reader.deleteLater()
        if self.preview_pending:
            self.preview_pending=False;QTimer.singleShot(0,lambda:self.refresh_preview(True) if not self.preview_closing else None)

    def closeEvent(self,event):
        if self.process.state()!=QProcess.ProcessState.NotRunning:self.close_pending=True;self.stop();event.ignore()
        elif self.map_page.reader is not None:
            self.map_page.reader.requestInterruption();self.map_page.reader.finished.connect(self.close);event.ignore()
        else:
            self.preview_timer.stop();self.preview_pending=False;self.preview_closing=True
            if self.frame_reader is not None:self.frame_reader.wait()
            self.resource_monitor.stop();event.accept()


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT);args=parser.parse_args(argv)
    # The OpenCV wheel injects Qt5 paths; this application uses PySide6 / Qt6.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
        if '/cv2/qt/' in os.environ.get(key,''):os.environ.pop(key)
    app=QApplication.instance() or QApplication(['lidar-annotator'])
    font_id=QFontDatabase.addApplicationFont(str(font_path('NotoSansCJKsc-Regular.otf')))
    if font_id>=0:app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0],10))
    app.setApplicationName('lidar-annotator')
    app.setDesktopFileName('lidar-annotator')
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window=AnnotationWindow(args.config);window.show();return app.exec()


if __name__=='__main__':raise SystemExit(main())
