"""Focused progress reporting and offscreen Qt progress text checks."""
import contextlib, io, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from tools.autolabel.task_progress import report_progress
from .cli import DEFAULT
from .fresh_pipeline import main
from .qt_app import AnnotationWindow, annotate_command, command_for


class ProgressTests(unittest.TestCase):
    def test_atomic_progress_is_quiet_when_redirected(self):
        with tempfile.TemporaryDirectory() as temp, patch('sys.stderr', io.StringIO()) as stderr:
            report_progress('semantic', 3, 971, frame=2, job=temp)
            report_progress('semantic', 4, 971, frame=3, job=temp)
            data = json.loads((Path(temp) / 'progress.json').read_text())
            self.assertEqual((data['completed'], data['total'], data['frame']), (4, 971, 3))
            self.assertEqual(stderr.getvalue(), '')
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_terminal_updates_same_line(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch('sys.stderr', stream):
            report_progress('semantic', 3, 10)
        self.assertIn('\r', stream.getvalue())
        self.assertIn('30.0%', stream.getvalue())
        self.assertNotIn('\n', stream.getvalue())

    def test_invalid_counters_rejected(self):
        with self.assertRaises(ValueError):
            report_progress('semantic', 11, 10)

    def test_gui_completion_prints_summary_not_command_json(self):
        report = dict(status='complete', elapsed_seconds=1.5, stages=[dict(stage='catalog', status='complete', commands=[['python', 'worker']])])
        output = io.StringIO()
        with patch('tools.lidar_annotator.fresh_pipeline.run', return_value=report), patch('tools.lidar_annotator.fresh_pipeline.detect_input_format', return_value='kitti'), contextlib.redirect_stdout(output):
            main(['--source', '/tmp/source', '--output', '/tmp/result', '--quiet-report'])
        text = output.getvalue()
        self.assertIn('处理完成', text)
        self.assertIn('/tmp/result/run_report.json', text)
        self.assertNotIn('commands', text)
        self.assertNotIn('"status"', text)
        self.assertIn('--quiet-report', annotate_command('/tmp/source', '/tmp/result', ['catalog'], 1))
        self.assertIn('--quiet-report', command_for(DEFAULT, ['catalog'], 1))


class ProgressTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, out, stages=None, input_format=None):
        window = AnnotationWindow(DEFAULT)
        window.active_config = {'output': str(out), **({'input_format': input_format} if input_format else {})}
        window.selected_stages = stages or []
        return window

    def test_stage_progress_text_includes_completed_count_and_frame_detail(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'run_report.json').write_text(json.dumps({'status': 'running', 'stages': [
                {'stage': 'semantic', 'status': 'complete', 'elapsed_seconds': 61.5}, {'stage': 'instances', 'status': 'running', 'started_unix': 100.0}]}))
            (out / 'progress.json').write_text(json.dumps({'stage': 'instances', 'completed': 4, 'total': 20, 'detail': '车辆实例', 'frame': 13}))
            window = self.make_window(out, ['semantic', 'instances', 'render'])
            try:
                with patch('tools.lidar_annotator.stage_timing.time.time', return_value=105.0):window.read_stage_log()
                self.assertIn('1 分 1.5 秒（完成）',window.stage_times.toolTip())
                self.assertIn('5.0 秒（进行中）',window.stage_times.toolTip())
                self.assertNotIn('\n',window.stage_times.text())
                self.assertFalse(window.stage_times.wordWrap())
                self.assertEqual(window.stage_times.minimumHeight(),window.stage_times.maximumHeight())
                self.assertIn('5.0 秒（进行中）',window.stage_times.text())
                from PySide6.QtWidgets import QDialog,QPlainTextEdit,QPushButton
                # Verify the button dispatch and copy action without opening a live window.
                with patch.object(QDialog,'show'),patch.object(QDialog,'raise_'),patch.object(QDialog,'activateWindow'):
                    window.stage_times_button.click()
                    dialog=window.findChild(QDialog,'stage_timing_dialog')
                    editor=dialog.findChild(QPlainTextEdit,'stage_timing_details')
                    self.assertTrue(editor.isReadOnly())
                    self.assertEqual(editor.toPlainText(),window.stage_times.toolTip())
                    editor.selectAll();self.assertEqual(editor.textCursor().selectedText().replace('\u2029','\n'),editor.toPlainText())
                    previous=QApplication.clipboard().text()
                    try:
                        dialog.findChild(QPushButton,'copy_stage_times').click()
                        self.assertEqual(QApplication.clipboard().text(),editor.toPlainText())
                    finally:QApplication.clipboard().setText(previous)
                    window.stage_times_button.click()
                    self.assertEqual(len(window.findChildren(QDialog,'stage_timing_dialog')),1)

                text = window.progress_text.text()
                for value in ('1/3', '行人 / 车辆实例标签', '4/20', '20.0%', '车辆实例', '帧 13'):
                    self.assertIn(value, text)
            finally:
                window.close()

    def test_opening_restores_current_job_timing_and_switching_clears_it(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);report=dict(status='complete',started_unix=100,elapsed_seconds=72.5,stages=[dict(stage='optimize',status='complete',elapsed_seconds=60.)])
            (out/'run_report.json').write_text(json.dumps(report))
            with patch.object(AnnotationWindow,'current_context',return_value={'output':str(out)}):
                window=AnnotationWindow(DEFAULT)
            try:
                self.assertIn('最近一次处理耗时：72.5 秒',window.stage_times.text())
                self.assertIn('位置优化与车辆去重：1 分 0.0 秒',window.stage_times.toolTip())
                self.assertNotIn('等待开始',window.stage_times.toolTip())
                window.load_saved_stage_times({'output':str(out/'other')})
                self.assertIn('暂无当前任务记录',window.stage_times.text())
                self.assertEqual(window.stage_times.toolTip(),'')
                report['stages']=[dict(stage='render',status='complete')]
                (out/'run_report.json').write_text(json.dumps(report))
                window.load_saved_stage_times({'output':str(out)})
                self.assertIn('旧报告未记录分阶段耗时',window.stage_times.toolTip())
                self.assertIn('72.5 秒',window.stage_times.text())
            finally:window.close()

    def test_raw_import_progress_is_shown_without_run_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'raw').mkdir()
            (out / 'raw' / 'status.json').write_text(json.dumps({'status': 'running', 'completed': 4, 'total': 12}))
            window = self.make_window(out, input_format='lmdb')
            try:
                window.read_stage_log()
                self.assertIn('读取 LMDB', window.progress_text.text())
                self.assertIn('4/12', window.progress_text.text())
            finally:
                window.close()

    def test_raw_complete_uses_imported_frame_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'raw').mkdir()
            (out / 'raw' / 'status.json').write_text(json.dumps({'status': 'complete', 'imported_frames': 8, 'total_recording_frames': 10}))
            window = self.make_window(out, input_format='lmdb')
            try:
                window.read_stage_log()
                self.assertIn('8/8', window.progress_text.text())
                self.assertNotIn('0/0', window.progress_text.text())
            finally:
                window.close()

    def test_start_resets_stale_progress_text(self):
        window = AnnotationWindow(DEFAULT)
        try:
            window.progress_text.setText('old stage text')
            context = {'output': '/tmp/nonexistent-progress-test', 'review_output': '/tmp/nonexistent-review', 'input_format': 'unknown'}
            with patch.object(window, 'current_context', return_value=context), patch('tools.lidar_annotator.qt_app.command_for', return_value=['worker']), patch.object(window.process, 'start'):
                window.start()
            self.assertEqual(window.progress_text.text(), '进度：准备开始')
        finally:
            window.close()

    def test_finished_sets_terminal_progress_text(self):
        window = AnnotationWindow(DEFAULT)
        try:
            window.active_config = None
            window.selected_stages = ['catalog']
            with patch.object(window, 'read_stdout'), patch.object(window, 'read_stage_log'):
                window.finished(0, None)
            self.assertIn('处理完成', window.progress_text.text())
            window.stopping = True
            with patch.object(window, 'read_stdout'), patch.object(window, 'read_stage_log'):
                window.finished(130, None)
            self.assertIn('已停止', window.progress_text.text())
            window.stopping = False
            with patch.object(window, 'read_stdout'), patch.object(window, 'read_stage_log'):
                window.finished(1, None)
            self.assertIn('失败', window.progress_text.text())
        finally:
            window.close()
