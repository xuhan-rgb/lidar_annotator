"""History persistence, legacy task discovery and editable selection controls."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
from .frame_history import FrameHistoryBox,frame_text,record_test,finish_test,test_history,history_path
from .qt_app import AnnotationWindow
from .cli import DEFAULT


class FrameHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_history_is_persistent_source_specific_and_keeps_distinct_result_runs(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'XDG_STATE_HOME':tmp}):
            root=Path(tmp);source=root/'recording';a=dict(output=str(root/'run1'),review_output=str(root/'run1/review'),input_format='lmdb');b=dict(a,output=str(root/'run2'))
            record_test(source,[0,1,2,9],a);record_test(source,[0,1,2,9],b);finish_test(a,'complete')
            entries=test_history(source)
            self.assertEqual([e['context']['output'] for e in entries],[b['output'],a['output']])
            self.assertEqual(entries[0]['frames_text'],'0-2 9');self.assertEqual(entries[1]['status'],'complete')
            self.assertEqual(test_history(root/'different-recording'),[])
            self.assertTrue(history_path().is_file());self.assertEqual(frame_text(None),'')

    def test_existing_temporary_jobs_are_discovered_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'XDG_STATE_HOME':tmp}):
            root=Path(tmp);source=root/'recording';debug=root/'jobs/debug'
            old=debug/'old';new=debug/'old_runs/new'
            for output in (old,new):
                output.mkdir(parents=True)
                (output/'job.json').write_text(json.dumps({'schema':'lidar_fresh_job_v1','source':str(source),'frame_ids':list(range(10))}))
                (output/'run_report.json').write_text(json.dumps({'status':'complete','started_unix':1 if output==old else 2}))
            with patch('tools.lidar_annotator.frame_history.directory',return_value=root/'jobs'):entries=test_history(source)
            self.assertEqual([e['context']['output'] for e in entries],[str(new),str(old)])
            self.assertEqual([e['frames_text'] for e in entries],['0-9','0-9'])
            self.assertFalse(history_path().exists())

    def test_dropdown_keeps_manual_input_and_normalizes_selected_labels(self):
        box=FrameHistoryBox();selected=[];box.entry_selected.connect(selected.append)
        try:
            entries=[dict(frames_text='0-9',started=1,status='complete',context={'output':'/tmp/first'}),dict(frames_text='0-9',started=2,status='failed',context={'output':'/tmp/second'})]
            box.setText('63 79');box.set_history(entries)
            self.assertEqual(box.text(),'63 79');self.assertEqual(box.count(),2)
            self.assertIn('已完成',box.itemText(0));self.assertIn('失败',box.itemText(1))
            box.setCurrentIndex(0);box.activated.emit(0)
            self.assertEqual(box.text(),'0-9');self.assertEqual(selected[-1]['context']['output'],'/tmp/first')
            box.clear();self.assertEqual(box.text(),'');self.assertEqual(box.count(),2)
            box.setText('20-30');self.assertEqual(box.text(),'20-30')
        finally:box.close()

    def test_long_history_entries_fit_a_narrow_sidebar(self):
        from PySide6.QtWidgets import QWidget,QFormLayout,QLabel
        parent=QWidget();parent.setFixedWidth(330);form=QFormLayout(parent)
        label=QLabel('处理帧（从 0 开始）');label.setMinimumWidth(145)
        box=FrameHistoryBox();box.set_history([dict(frames_text='0-9 63 79 824',started=1,status='complete',context={'output':'/tmp/test'})])
        form.addRow(label,box)
        try:
            self.assertLessEqual(parent.minimumSizeHint().width(),330)
            self.assertLessEqual(box.minimumSizeHint().width(),110)
            self.assertIn('已完成',box.itemText(0))
            box.setText('0-9');self.assertEqual(box.text(),'0-9')
        finally:parent.close()

    def test_selecting_history_loads_exact_old_run_without_starting_a_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'XDG_STATE_HOME':tmp}):
            root=Path(tmp);source=root/'recording';old=dict(output=str(root/'old'),review_output=str(root/'old/review'),input_format='lmdb')
            record_test(source,list(range(10)),old,'complete')
            window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_root.setText(str(source));window.refresh_test_history()
                self.assertEqual(window.frames.count(),1)
                with patch.object(window,'refresh_preview'),patch.object(window.process,'start') as start:
                    window.frames.setCurrentIndex(0);window.frames.activated.emit(0)
                self.assertEqual(window.frames.text(),'0-9');self.assertEqual(window.current_context()['output'],old['output'])
                start.assert_not_called();self.assertFalse(hasattr(window,'frames_help'))
            finally:window.close()

    def test_execution_is_recorded_and_preview_is_not(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'XDG_STATE_HOME':tmp}):
            root=Path(tmp);window=AnnotationWindow(DEFAULT)
            try:
                window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_root.setText(str(root/'recording'));window.lmdb_output.setText(str(root/'output'))
                with patch.object(window.process,'start'):window.start(True)
                window.process.started.emit();self.assertFalse(history_path().exists())
                with patch.object(window,'read_stdout'):window.finished(0,QProcess.ExitStatus.NormalExit)
                with patch.object(window.process,'start'):window.start()
                window.process.started.emit()
                self.assertEqual(test_history(root/'recording')[0]['status'],'running')
                with patch.object(window,'read_stdout'):window.finished(0,QProcess.ExitStatus.NormalExit)
                self.assertEqual(test_history(root/'recording')[0]['status'],'complete')
                self.assertEqual(window.frames.text(),'')
            finally:window.close()
