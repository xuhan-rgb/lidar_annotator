"""Full regeneration removes only verified results for the same task."""
import contextlib,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
from .cli import DEFAULT,STAGES
from .qt_app import AnnotationWindow
from .fresh_pipeline import run
from .test_lmdb_input import fixture


class RegenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def window(self,root):
        source=fixture(root);window=AnnotationWindow(DEFAULT)
        window.input_format.setCurrentIndex(window.input_format.findData('lmdb'));window.lmdb_root.setText(str(source));window.lmdb_output.setText(str(root/'previous'))
        window.select_stages(['render','catalog'])
        return window

    def test_checked_regeneration_forces_all_stages_and_all_classes_in_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                old=Path(window.current_context()['output']);old.mkdir(parents=True);(old/'old-label.npz').write_bytes(b'old result')
                (old/'job.json').write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=window.lmdb_root.text(),frame_ids=[0])))
                window.refresh_semantic.setChecked(True);window.refresh_classes.setCurrentIndex(2)
                window.regenerate_all.setChecked(True)
                self.assertEqual([s for s,c in window.checks.items() if c.isChecked()],list(STAGES))
                self.assertFalse(window.refresh_semantic.isEnabled());self.assertFalse(window.refresh_instances.isEnabled());self.assertFalse(window.refresh_classes.isEnabled())
                with patch.object(window.process,'start') as start:window.start()
                args=start.call_args.args[1];new=Path(args[args.index('--work-dir')+1])
                self.assertNotEqual(new,old);self.assertEqual(new.parent,old.parent/(old.name+'_runs'))
                self.assertIn('--refresh-semantic',args);self.assertIn('--refresh-instances',args);self.assertNotIn('--semantic-classes',args)
                self.assertEqual(args[args.index('--stages')+1:args.index('--workers')],list(STAGES))
                self.assertEqual(window.current_context()['output'],str(new));self.assertEqual(window.active_config['output'],str(new))
                self.assertTrue(window.regenerate_all.isChecked());self.assertFalse(window.regenerate_all.isEnabled())
                self.assertFalse(old.exists())
                self.assertIsNone(window.camera_view.data)
                # The regular CLI gets a completely empty job and plans all stages.
                plan=run(window.lmdb_root.text(),new,dry_run=True,refresh_semantic=True,refresh_instances=True)
                self.assertEqual([s['stage'] for s in plan['plan']],list(STAGES));self.assertFalse(new.exists())
                window.finished(1,QProcess.ExitStatus.NormalExit)
                self.assertEqual(window.current_context()['output'],str(new));self.assertTrue(window.regenerate_all.isEnabled())
                self.assertTrue(window.regenerate_all.isChecked())
                window.resume_cached.setChecked(True);self.assertFalse(window.regenerate_all.isChecked())
                new.mkdir(parents=True)
                public=Path(window.active_config['dataset_output']);public.mkdir();(public/'metadata.npz').touch()
                with patch('tools.lidar_annotator.qt_app.QDesktopServices.openUrl') as opened:
                    window.open_result();self.assertEqual(opened.call_args.args[0].toLocalFile(),str(public))
            finally:window.close()

    def test_current_recipe_inside_an_old_result_directory_is_preserved(self):
        from .recipe_config import default_recipe
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                old=Path(window.current_context()['output']);old.mkdir(parents=True)
                (old/'job.json').write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=window.lmdb_root.text(),frame_ids=[0])))
                recipe=default_recipe();recipe['raw_classes'][1]['name']='custom person'
                (old/'recipe.json').write_text(json.dumps(recipe));window.profile.setText(str(old/'recipe.json'))
                window.regenerate_all.setChecked(True)
                self.assertEqual(window.regenerate_all.text(),'重新生成全部结果')
                with patch.object(window.process,'start') as start:window.start()
                args=start.call_args.args[1];saved=Path(args[args.index('--recipe')+1])
                self.assertFalse(old.exists());self.assertTrue(saved.is_file())
                self.assertEqual(json.loads(saved.read_text()),recipe)
            finally:window.close()

    def test_preview_does_not_consume_checkbox_or_change_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                old=window.current_context()['output'];window.regenerate_all.setChecked(True)
                with patch.object(window.process,'start') as start:window.start(True)
                args=start.call_args.args[1];self.assertIn('--dry-run',args)
                self.assertTrue(window.regenerate_all.isChecked());self.assertEqual(window.current_context()['output'],old)
                self.assertFalse((root/'previous_runs').exists())
            finally:window.close()

    def test_latest_result_is_restored_after_reopening_for_the_same_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                window.regenerate_all.setChecked(True)
                with patch.object(window.process,'start'):window.start()
                new=window.current_context()['output'];source=window.lmdb_root.text()
            finally:window.close()
            restored=AnnotationWindow(DEFAULT)
            try:
                restored.input_format.setCurrentIndex(restored.input_format.findData('lmdb'));restored.lmdb_root.setText(source);restored.lmdb_output.setText(str(root/'previous'))
                self.assertEqual(restored.current_context()['output'],new)
                self.assertFalse(restored.regenerate_all.isChecked())
                with patch.object(restored.process,'start') as start:restored.start()
                args=start.call_args.args[1]
                self.assertEqual(args[args.index('--work-dir')+1],new)
                self.assertNotIn('--refresh-semantic',args)
            finally:restored.close()

    def test_invalid_command_does_not_replace_the_latest_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                window.lmdb_root.clear();window.regenerate_all.setChecked(True)
                with patch.object(window.process,'start') as start:window.start()
                start.assert_not_called();self.assertTrue(window.regenerate_all.isChecked())
                self.assertFalse((root/'previous_runs/latest.json').exists())
            finally:window.close()

    def test_latest_pointer_cannot_redirect_into_another_results_directory(self):
        from .result_runs import latest_result,remember_result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base={'output':str(root/'previous')}
            other={'output':str(root/'outside'),'review_output':str(root/'outside/review'),'input_format':'lmdb'}
            remember_result(base,root/'source',None,other)
            self.assertEqual(latest_result(base,root/'source',None),base)

    def test_unchecked_uses_existing_directory_without_forcing_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            window=self.window(Path(tmp))
            try:
                old=window.current_context()['output']
                with patch.object(window.process,'start') as start:window.start()
                args=start.call_args.args[1]
                self.assertEqual(args[args.index('--work-dir')+1],old)
                self.assertNotIn('--refresh-semantic',args);self.assertNotIn('--refresh-instances',args)
                self.assertEqual(args[args.index('--stages')+1:args.index('--workers')],['render','catalog'])
            finally:window.close()

    def test_subset_result_context_survives_finish_and_is_specific_to_source_and_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                window.frames.setText('0');old=window.current_context()['output'];window.regenerate_all.setChecked(True)
                with patch.object(window.process,'start'):window.start()
                new=window.current_context()['output'];self.assertNotEqual(old,new)
                window.finished(0,QProcess.ExitStatus.NormalExit)
                self.assertEqual(window.current_context()['output'],new)
                window.frames.clear();self.assertNotEqual(window.current_context()['output'],new)
                window.frames.setText('0');self.assertEqual(window.current_context()['output'],new)
                window.lmdb_root.setText(str(root/'another-source'));self.assertNotEqual(window.current_context()['output'],new)
            finally:window.close()

    def test_each_explicit_regeneration_is_a_distinct_run_and_workers_receive_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                outputs=[]
                for _ in range(2):
                    window.regenerate_all.setChecked(True)
                    with patch.object(window.process,'start'):window.start()
                    outputs.append(window.current_context()['output']);window.finished(1,QProcess.ExitStatus.NormalExit)
                self.assertNotEqual(*outputs)
                with contextlib.redirect_stdout(io.StringIO()),patch('tools.lidar_annotator.fresh_pipeline.execute') as execute:
                    result=run(window.lmdb_root.text(),outputs[-1],refresh_semantic=True,refresh_instances=True)
                self.assertEqual(result['status'],'complete');self.assertEqual(len(execute.call_args_list),len(STAGES))
                for call in execute.call_args_list:
                    args=call.args[0];stage=args[args.index('--stage')+1]
                    self.assertEqual('--refresh' in args,stage in ('semantic','instances'))
            finally:window.close()
