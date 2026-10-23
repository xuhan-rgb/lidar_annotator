"""Execution modes preserve inference while explicitly refreshing downstream stages."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import test_regeneration
from .fresh_pipeline import run, stage
from .qt_app import TEMPORAL_FLOW, EXPORT_FLOW
from .recipe_config import default_recipe
import json


class ExecutionModeTests(unittest.TestCase):
    setUpClass=classmethod(test_regeneration.RegenerationTests.setUpClass.__func__)
    window=test_regeneration.RegenerationTests.window

    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);root=Path(directory.name)
        environment=patch.dict('os.environ',{'XDG_CONFIG_HOME':str(root/'config'),
                                           'XDG_STATE_HOME':str(root/'state'),'LIDAR_DATA_ROOT':str(root/'data')})
        environment.start();self.addCleanup(environment.stop)

    def test_startup_postprocess_keeps_saved_recipe_matching_existing_job(self):
        from .recipe_config import save_system_recipe,load_recipe
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{
            'XDG_CONFIG_HOME':str(Path(tmp)/'config'),'XDG_STATE_HOME':str(Path(tmp)/'state'),
            'LIDAR_DATA_ROOT':str(Path(tmp)/'data')}):
            root=Path(tmp);recipe=default_recipe();recipe['raw_classes'][1]['name']='saved person'
            saved=save_system_recipe(recipe)
            (saved.parent/'ui.ini').write_text('[General]\nexecution_mode=postprocess\n')
            window=self.window(root)
            try:
                self.assertTrue(window.recompute_postprocess.isChecked())
                job=Path(window.current_context()['output'])
                for name in ('data/labels/zgc/index.json','data/person/index.json','data/vehicle/index.json'):
                    path=job/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
                (job/'recipe.json').write_text(json.dumps(recipe))
                with patch.object(window.process,'start') as start:window.start(True)
                args=start.call_args.args[1];selected_recipe=Path(args[args.index('--recipe')+1])
                self.assertEqual(selected_recipe,saved)
                self.assertEqual(load_recipe(selected_recipe),recipe)
                plan=run(window.lmdb_root.text(),job,dry_run=True,refresh_postprocess=True,
                         recipe_path=selected_recipe,dataset_output=window.active_config['dataset_output'])
                self.assertEqual([row['stage'] for row in plan['plan']],list(TEMPORAL_FLOW+EXPORT_FLOW))
            finally:window.close()
    def test_postprocess_mode_is_exclusive_and_preserves_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            window=self.window(Path(tmp))
            try:
                old=window.current_context()['output']
                self.assertTrue(window.resume_cached.isChecked())
                window.regenerate_all.setChecked(True)
                window.recompute_postprocess.setChecked(True)
                self.assertFalse(window.regenerate_all.isChecked())
                self.assertFalse(window.resume_cached.isChecked())
                self.assertFalse(window.refresh_semantic.isEnabled())
                self.assertFalse(window.refresh_instances.isEnabled())
                with patch.object(window.process,'start') as start,patch('tools.lidar_annotator.qt_app.clear_previous_results') as clear:
                    window.start(True)
                clear.assert_not_called()
                args=start.call_args.args[1]
                self.assertIn('--refresh-postprocess',args)
                self.assertNotIn('--refresh-semantic',args)
                self.assertNotIn('--refresh-instances',args)
                self.assertEqual(args[args.index('--stages')+1:args.index('--workers')],list(TEMPORAL_FLOW+EXPORT_FLOW))
                self.assertEqual(window.current_context()['output'],old)
            finally:window.close()

    def test_postprocess_plan_never_schedules_models_and_forces_every_downstream_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                job=Path(window.current_context()['output'])
                for name in ('data/labels/zgc/index.json','data/person/index.json','data/vehicle/index.json'):
                    p=job/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
                (job/'recipe.json').write_text(json.dumps(default_recipe()))
                plan=run(window.lmdb_root.text(),job,dry_run=True,refresh_postprocess=True)
                self.assertEqual([r['stage'] for r in plan['plan']],list(TEMPORAL_FLOW+EXPORT_FLOW))
                for row in plan['plan']:
                    self.assertIn('--refresh',row['commands'][0])
                    with patch('tools.lidar_annotator.fresh_pipeline.config',return_value={}),patch('tools.lidar_annotator.fresh_pipeline._stage') as execute,patch('tools.lidar_annotator.stage_cache.complete'),patch('tools.lidar_annotator.stage_cache.reuse',side_effect=lambda job,name,force: not force) as reuse:
                        stage(job,row['stage'],1,refresh=True)
                    execute.assert_called_once_with(job,row['stage'],1,True)
                    reuse.assert_called_once_with(job,row['stage'],force=True)
                self.assertEqual(json.loads((job/'recipe.json').read_text()),default_recipe())
            finally:window.close()

    def test_missing_inference_and_changed_recipe_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);window=self.window(root)
            try:
                job=Path(window.current_context()['output'])
                with self.assertRaisesRegex(ValueError,'缺少已有'):
                    run(window.lmdb_root.text(),job,dry_run=True,refresh_postprocess=True)
                for name in ('data/labels/zgc/index.json','data/person/index.json','data/vehicle/index.json'):
                    p=job/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
                (job/'recipe.json').write_text(json.dumps(default_recipe()))
                recipe=default_recipe();recipe['raw_classes'][1]['name']='changed'
                changed=root/'changed.json';changed.write_text(json.dumps(recipe))
                with self.assertRaisesRegex(ValueError,'配置已改变'):
                    run(window.lmdb_root.text(),job,dry_run=True,refresh_postprocess=True,recipe_path=changed)
            finally:window.close()

    def test_clicked_mode_is_restored_after_reopening(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'XDG_CONFIG_HOME':tmp}):
            for name in ('recompute_postprocess','regenerate_all','resume_cached'):
                window=test_regeneration.AnnotationWindow(test_regeneration.DEFAULT)
                try:
                    getattr(window,name).click()
                    window.ui_settings.sync()
                finally:window.close()
                restored=test_regeneration.AnnotationWindow(test_regeneration.DEFAULT)
                try:
                    self.assertTrue(getattr(restored,name).isChecked())
                    self.assertEqual(sum(b.isChecked() for b in restored.execution_modes.buttons()),1)
                    if name=='recompute_postprocess':
                        self.assertFalse(restored.refresh_semantic.isEnabled())
                        self.assertEqual([k for k,v in restored.checks.items() if v.isChecked()],list(TEMPORAL_FLOW+EXPORT_FLOW))
                finally:restored.close()
