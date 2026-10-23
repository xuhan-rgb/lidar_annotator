import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from .cli import STAGES,commands,load_config,run


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        backend=root/'backend.json';backend.write_text(json.dumps({'temporal_output':str(root/'temporal')}))
        profile=root/'profile.json';profile.write_text(json.dumps(dict(adapter='zgc_cached_v1',backend_config=str(backend),python=sys.executable,sam_python=sys.executable,sam_source=str(root/'sam'),semantic_runner=str(root/'runner.py'),output=str(root/'output'))))
        self.config=load_config(profile)

    def test_default_keeps_visual_and_semantic_caches(self):
        for stage in ('semantic','instances'):
            plan=commands(self.config,stage)
            self.assertEqual(len(plan),1)
            self.assertIn('check',plan[0])
        self.assertEqual(STAGES[:5],('semantic','instances','temporal','tracking','optimize'))

    def test_tracking_uses_sam_environment_and_production_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.config['backend']['temporal_output']=tmp
            plan=commands(self.config,'tracking')
            self.assertEqual(plan[0][0],self.config['sam_python'])
            self.assertTrue(plan[1][1].endswith('prepare_person_observations.py'))

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.config['output']=str(Path(tmp)/'absent')
            result=run(self.config,STAGES,dry_run=True)
            self.assertEqual(result['status'],'dry_run')
            self.assertFalse(Path(self.config['output']).exists())

    def test_completed_video_cache_keeps_existing_visual_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'map_review/sam3_full_video';video.mkdir(parents=True)
            (video/'report.json').write_text(json.dumps({'status':'complete'}))
            (video/'object_category_audit.json').write_text(json.dumps({'excluded_objects':[{'sam_video_id':7}]}))
            self.config['backend']['temporal_output']=tmp
            plan=commands(self.config,'tracking')
            self.assertIn('check',plan[0])
            self.assertFalse(any(value.endswith('run_sam3_full_video.py') for value in plan[0]))

    def test_partial_video_cannot_reuse_manual_id_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'map_review/sam3_full_video';video.mkdir(parents=True)
            (video/'object_category_audit.json').write_text(json.dumps({'excluded_objects':[{'sam_video_id':7}]}))
            self.config['backend']['temporal_output']=tmp
            with self.assertRaises(ValueError):commands(self.config,'tracking')

    def test_partial_frames_cannot_change_sequence_optimization(self):
        with self.assertRaises(ValueError):run(self.config,['optimize'],frames=[79],dry_run=True)
        plan=commands(self.config,'render',frames=[79])
        self.assertEqual(plan[0][-2:],['--frames','79'])

    def test_failed_stage_stops_before_next_and_saves_status(self):
        calls=[]
        def fail(command,**kwargs):
            calls.append(command)
            raise subprocess.CalledProcessError(1,command)
        with tempfile.TemporaryDirectory() as tmp:
            self.config['output']=tmp
            with self.assertRaises(subprocess.CalledProcessError):run(self.config,['semantic','instances'],runner=fail)
            report=json.loads((Path(tmp)/'run_report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(len(calls),1)
            self.assertEqual(report['stages'][0]['stage'],'semantic')


if __name__=='__main__':unittest.main()
