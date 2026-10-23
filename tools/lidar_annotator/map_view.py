"""Interactive whole-task map; loading and exporting run outside the UI thread."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QMenu, QToolButton,
    QProgressBar, QPushButton, QVBoxLayout, QWidget)

from .optimized_map import load_optimized_map, box_corners, person_trajectory_segments, person_track_starts
from .display_range import DEFAULT_LIMITS
from .live_preview import semantic_legend_html
from .cli import REPO


def _map_context_key(context):
    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else REPO / path

    output = absolute(context['output'])
    backend = context.get('backend', {})
    temporal = absolute(backend.get('temporal_output', output / 'temporal'))
    source_index = absolute(backend.get('source_index', output / 'data/labels/zgc/index.json'))
    pose_npz = absolute(backend.get('pose_npz', temporal / 'poses.npz'))
    tracks = temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks'
    vehicles = temporal / 'map_review/history_fit/bev_dedup'
    paths = (source_index, pose_npz, vehicles / 'vehicle_map_boxes.npz', tracks / 'centers_only.npz',
             tracks / 'final_centers.npz', tracks / 'report.json', output / 'recipe.json', absolute(backend.get('person_root', output / 'data/person')) / 'index.json')
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        except FileNotFoundError:
            stamps.append((str(path.resolve()), None))
    return context['output'], tuple(context.get('map_limits', DEFAULT_LIMITS)), tuple(stamps)


class MapReader(QThread):
    loaded = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, context, spacing, parent=None):
        super().__init__(parent); self.context = context; self.spacing = spacing

    def run(self):
        try:
            data = load_optimized_map(self.context, self.spacing, self.progress.emit, self.isInterruptionRequested)
            if not self.isInterruptionRequested():self.loaded.emit(data)
        except InterruptedError:pass
        except Exception as exc:self.failed.emit(str(exc))


class MapCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None; self.center = np.zeros(2); self.zoom = 1.; self.pan = QPointF(); self.drag = None
        self.show_people = True; self.show_cars = True; self.show_path = True
        self.show_start = True; self.show_person_paths = True; self.show_frames = True
        self.setMouseTracking(True)
        self.setMinimumSize(250, 220)

    def set_map(self, data):
        self.data = data
        bounds = [data['xyz'][:, :2], data['person_xyz'][:, :2], data['trajectory'][:, :2]]
        bounds += [box_corners(box)[:, :2] for box in data['boxes']]
        points = np.concatenate(bounds)
        low, high = (points.min(0), points.max(0)) if len(points) else (np.array([-20, -20]), np.array([20, 20]))
        middle = (low + high) / 2; self.center = np.array([-middle[1], -middle[0]])
        self.extent = max(float(np.ptp(points, axis=0).max()) / 2, 1.) * 1.08 if len(points) else 20.
        self.reset_view()

    def reset_view(self):
        self.zoom = 1.; self.pan = QPointF(); self.update()

    def scale(self):return min(self.width(), self.height()) / (2 * self.extent) * self.zoom

    def project(self, xy):
        return (np.c_[-xy[:, 1], -xy[:, 0]] - self.center) * self.scale() + np.array([self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y()])

    def paintEvent(self, event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor('#14171c'))
        if self.data is None:
            painter.setPen(Qt.GlobalColor.white); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '正在加载全帧地图…'); return
        # Rasterize points in NumPy; avoid millions of Python/QPointF objects.
        w, h = self.width(), self.height()
        pixels = np.floor(self.project(self.data['xyz'][:, :2])).astype(np.int32)
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        labels = self.data['labels'][valid]; pixels = pixels[valid]
        flat = pixels[:, 1] * w + pixels[:, 0]
        order = [255, 0, 3, 4, 5, 6, 7, 8, 9, 2, 1]
        ranks = np.zeros(256, np.uint8)
        for rank, cid in enumerate(order, 1):ranks[cid] = rank
        image = np.zeros(w * h, np.uint8)
        np.maximum.at(image, flat, ranks[labels])
        colors = np.full((len(order) + 1, 3), [20, 23, 28], np.uint8)
        palette = {c['id']: c['rgb'] for c in self.data['palette']}
        for rank, cid in enumerate(order, 1):colors[rank] = palette.get(cid, [100, 100, 100])
        rgb = np.ascontiguousarray(colors[image].reshape(h, w, 3))
        painter.drawImage(QPointF(0, 0), QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888))
        # Grid labels are map coordinates, with +X up and +Y left.
        step = max(1., 10 ** np.floor(np.log10(2 * self.extent / self.zoom / 8)))
        if (2 * self.extent / self.zoom) / step > 16:step *= 5
        origin = np.array([w / 2 + self.pan.x(), h / 2 + self.pan.y()])
        world_low = (np.array([0, 0]) - origin) / self.scale() + self.center
        world_high = (np.array([w, h]) - origin) / self.scale() + self.center
        painter.setPen(QPen(QColor('#343a40'), 1))
        for axis in (0, 1):
            for value in np.arange(np.ceil(world_low[axis] / step) * step, world_high[axis], step):
                pixel = (value - self.center[axis]) * self.scale() + origin[axis]
                if axis == 0:
                    painter.drawLine(QPointF(pixel, 0), QPointF(pixel, h)); text_pos = QPointF(pixel + 3, h - 8)
                else:
                    painter.drawLine(QPointF(0, pixel), QPointF(w, pixel)); text_pos = QPointF(5, pixel - 3)
                painter.setPen(QColor('#aab6c2')); painter.drawText(text_pos, f'{"Y" if axis == 0 else "X"} {-value:g} m'); painter.setPen(QColor('#343a40'))
        if self.show_path:
            painter.setPen(QPen(QColor('#eeeeee'), 1))
            for segment in np.unique(self.data['segments']):
                path = self.project(self.data['trajectory'][self.data['segments'] == segment, :2])
                painter.drawPolyline(QPolygonF([QPointF(*p) for p in path]))
        if self.show_frames:
            painter.setPen(QPen(QColor('#ffffff'), 1))
            for position in self.project(self.data['trajectory'][:, :2]):
                painter.drawEllipse(QPointF(*position), 2.5, 2.5)
        if self.show_start and len(self.data['trajectory']):
            painter.setPen(QPen(QColor('#ffffff'), 2))
            point = QPointF(*self.project(self.data['trajectory'][:1, :2])[0])
            painter.drawEllipse(point, 5, 5)
        if self.show_cars:
            painter.setPen(QPen(QColor('#ffdc00'), 2))
            for box, tid in zip(self.data['boxes'], self.data['car_ids']):
                pixels = self.project(box_corners(box)[:4, :2]); painter.drawPolygon(QPolygonF([QPointF(*p) for p in pixels]))
                painter.drawText(QPointF(*pixels[0]) + QPointF(4, -4), f'V{tid}')
        if self.show_person_paths:
            painter.setPen(QPen(QColor('#00ffff'), 2))
            for tid, indices in person_trajectory_segments(self.data):
                pixels = self.project(self.data['person_xyz'][indices, :2])
                painter.drawPolyline(QPolygonF([QPointF(*p) for p in pixels]))
        if self.show_people:
            painter.setPen(QPen(QColor('#00ffff'), 2))
            track_only=self.data.get('person_track_only',np.zeros(len(self.data['person_xyz']),bool))
            for tid in np.unique(self.data['person_ids']):
                ix = np.flatnonzero(self.data['person_ids'] == tid)
                pixels = self.project(self.data['person_xyz'][ix, :2])
                for index,point in zip(ix,pixels):
                    if track_only[index]:
                        painter.setPen(QPen(QColor('#ffbf66'),2));p=QPointF(*point)
                        painter.drawLine(p+QPointF(-3,-3),p+QPointF(3,3));painter.drawLine(p+QPointF(-3,3),p+QPointF(3,-3))
                    else:painter.setPen(QPen(QColor('#00ffff'),2));painter.drawEllipse(QPointF(*point),3,3)
                painter.setPen(QPen(QColor('#00ffff'),2))
                last = np.argmax(self.data['person_frames'][ix])
                painter.drawText(QPointF(*pixels[last]) + QPointF(4, -4), f'P{tid}')
            painter.setBrush(QColor('#00ffff'))
            for tid, index in person_track_starts(self.data):
                point = QPointF(*self.project(self.data['person_xyz'][[index], :2])[0])
                painter.drawPolygon(QPolygonF([point + QPointF(0, -6), point + QPointF(-5, 4), point + QPointF(5, 4)]))
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def wheelEvent(self, event):
        if self.data is None:return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        previous = self.zoom; self.zoom = min(100., max(.05, self.zoom * factor))
        factor = self.zoom / previous
        anchor = event.position() - QPointF(self.width() / 2, self.height() / 2)
        self.pan = anchor - (anchor - self.pan) * factor; self.update(); event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:self.drag = event.position()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            self.pan += event.position() - self.drag; self.drag = event.position(); self.update()
        elif self.data is not None:
            cursor = np.array([event.position().x(), event.position().y()]); tooltip = ''
            if self.show_people:
                for tid, index in person_track_starts(self.data):
                    point = self.project(self.data['person_xyz'][[index], :2])[0]
                    if np.linalg.norm(point - cursor) <= 8:
                        tooltip = f"行人 P{tid} 起点 · 起始帧 {self.data['person_frames'][index]}"; break
            if not tooltip and self.show_frames:
                positions = self.project(self.data['trajectory'][:, :2])
                nearby = np.flatnonzero(np.linalg.norm(positions - cursor, axis=1) <= 8)
                tooltip = '\n'.join(f"帧 {self.data['trajectory_frames'][i]} · X {self.data['trajectory'][i,0]:.2f} / Y {self.data['trajectory'][i,1]:.2f} / Z {self.data['trajectory'][i,2]:.2f} m" for i in nearby)
            if not tooltip and self.show_start and len(self.data['trajectory']):
                position = self.project(self.data['trajectory'][:1, :2])[0]
                if np.linalg.norm(position - cursor) <= 8:
                    xyz = self.data['trajectory'][0]
                    tooltip = f"起始帧 {self.data['trajectory_frames'][0]} · X {xyz[0]:.2f} / Y {xyz[1]:.2f} / Z {xyz[2]:.2f} m"
            self.setToolTip(tooltip)

    def mouseReleaseEvent(self, event):self.drag = None


class MapPage(QWidget):

    def __init__(self, context_provider, parent=None):
        super().__init__(parent); self.context_provider = context_provider; self.reader = None; self.context_key = None; self.html = None
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 2, 0, 0); layout.setSpacing(4); toolbar = QHBoxLayout(); toolbar.setSpacing(6)
        self.reload = QPushButton('刷新地图'); self.reload.clicked.connect(self.refresh)
        self.spacing = QDoubleSpinBox(); self.spacing.setRange(0, 1); self.spacing.setSingleStep(.05); self.spacing.setValue(.15); self.spacing.setSuffix(' m'); self.spacing.setSpecialValueText('全部点')
        self.spacing.setToolTip('仅控制地图显示的体素合并间距；0 显示全部点，完整点云仍保存在地图归档中。')
        self.open_3d = QPushButton('3D 地图'); self.open_3d.setEnabled(False); self.open_3d.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.html)))
        self.open_3d.setToolTip('在浏览器中打开交互 3D 地图')
        self.canvas = MapCanvas()
        reset = QPushButton('重置视图'); reset.clicked.connect(self.canvas.reset_view)
        self.display_button = QToolButton(); self.display_button.setText('显示内容')
        self.display_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.display_button); self.display_button.setMenu(menu); self.display_actions = {}
        for text, name in (('行人位置标记', 'show_people'), ('车辆框', 'show_cars'), ('行人中心轨迹', 'show_person_paths'), ('采集轨迹', 'show_path'), ('每帧位置', 'show_frames'), ('起始帧位置', 'show_start')):
            action = QAction(text, menu); action.setCheckable(True); action.setChecked(True); menu.addAction(action)
            action.toggled.connect(lambda value, n=name: (setattr(self.canvas, n, value), self.canvas.update()))
            self.display_actions[name] = action
        for widget in (self.reload, QLabel('体素'), self.spacing, self.open_3d, reset, self.display_button):toolbar.addWidget(widget)
        toolbar.addStretch(); layout.addLayout(toolbar)
        self.legend = QLabel(); self.legend.setWordWrap(True); self.legend.hide(); layout.addWidget(self.legend)
        layout.addWidget(self.canvas, 1)
        self.progress = QProgressBar(); self.progress.hide(); layout.addWidget(self.progress)
        self.status = QLabel('全帧优化后查看累积点云、行人位置和车辆框。滚轮缩放，左键拖动。'); self.status.setWordWrap(True); layout.addWidget(self.status)

    def matches_context(self, context):
        return self.context_key == _map_context_key(context)

    def refresh(self):
        if self.reader is not None:return
        try:
            context = self.context_provider(); self.context_key = _map_context_key(context)
            self.canvas.data = None; self.canvas.update(); self.open_3d.setEnabled(False)
            self.reader = MapReader(context, self.spacing.value(), self)
            self.reader.loaded.connect(self.loaded); self.reader.failed.connect(lambda error: self.status.setText(error))
            self.reader.progress.connect(self.update_progress); self.reader.finished.connect(self.loading_finished)
            self.reload.setEnabled(False); self.spacing.setEnabled(False)
            self.progress.setRange(0, 0); self.progress.show(); self.status.setText('正在读取已保存地图；数据变化时自动更新…'); self.reader.start()
        except (OSError, ValueError, KeyError) as exc:self.status.setText(str(exc))

    def update_progress(self, done, total):
        self.progress.setRange(0, total); self.progress.setValue(done)
        self.status.setText(f'累计 {done}/{total} 帧' + ('；正在保存完整点云并生成地图…' if done == total else ''))

    def loaded(self, data):
        try:
            if not self.matches_context(self.context_provider()):
                self.status.setText('数据来源已切换，请重新加载当前任务的地图。');return
        except (OSError, ValueError, KeyError) as exc:self.status.setText(str(exc));return
        self.canvas.set_map(data); self.html = data['html']; self.open_3d.setEnabled(True)
        self.legend.setText(semantic_legend_html(data['palette']))
        self.legend.setText(self.legend.text() + '　<span style="color:#00ffff">○</span> 有当前点支持的位置　<span style="color:#ffbf66">×</span> 仅轨迹估计　<span style="color:#00ffff">─ / ▲</span> 轨迹 / 起点（独立于语义点云）')
        self.legend.show()
        extent, zmin, zmax = data['limits']
        origin = '已读取保存的地图' if data.get('cache_reused') else '地图已生成并保存'
        self.status.setText(f"{origin} · 当前任务 {data['frame_count']} 帧 · 累计 {data['point_count']:,} 点 · 显示 {len(data['xyz']):,} 点 · 行人 {len(np.unique(data['person_ids']))} 个 · 车辆 {len(data['car_ids'])} 个")
        self.status.setText(self.status.text()+(' · 行人：全局参考位置' if data.get('person_position_policy')=='global_reference_xy_v1' else ' · 行人：旧射线位置，执行流程后更新')+f" · 仅轨迹估计 {int(np.asarray(data.get('person_track_only',[])).sum())} 条")
        self.status.setToolTip(f"与联合标注共用语义和范围：XY ±{extent:g} m、Z [{zmin:g},{zmax:g}] m；地图按记录位姿累积。\n完整点云：{data['archive']}")

    def loading_finished(self):
        self.reader.deleteLater(); self.reader = None; self.progress.hide()
        self.reload.setEnabled(True); self.spacing.setEnabled(True)
