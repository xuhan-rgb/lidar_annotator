import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .stage_timing import measure_stage,stage_timing_text
from . import cli,fresh_pipeline
from .test_lmdb_input import fixture


class StageTimingTests(unittest.TestCase):
    def test_live_completed_and_legacy_times(self):
        rows=[dict(stage='semantic',status='complete',elapsed_seconds=125.25),dict(stage='render',status='running',started_unix=100),dict(stage='catalog',status='complete')]
        text=stage_timing_text(rows,{'semantic':'语义'},now=105)
        self.assertIn('语义：2 分 5.2 秒（完成）',text)
        self.assertIn('render：5.0 秒（进行中）',text)
        self.assertNotIn('catalog',text)
        self.assertEqual(stage_timing_text([],{}),'阶段耗时：等待开始')

    def test_duration_uses_monotonic_clock_even_when_system_clock_changes(self):
        entry={}
        with patch('tools.lidar_annotator.stage_timing.time.perf_counter',side_effect=[10,12.5]),patch('tools.lidar_annotator.stage_timing.time.time',side_effect=[100,90]):
            with measure_stage(entry):pass
        self.assertEqual(entry['elapsed_seconds'],2.5)

    def test_reports_save_each_stage_time_in_success_failure_and_cancellation(self):
        for backend in ('fresh','legacy'):
            for failure,state in ((None,'complete'),(RuntimeError('failed'),'failed'),(KeyboardInterrupt(),'cancelled')):
                with self.subTest(backend=backend,state=state),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);job=root/'job';source=fixture(root)
                    calls=[]
                    with patch('tools.lidar_annotator.stage_timing.time.perf_counter',return_value=10.) as clock:
                        def execute(*args,**kwargs):
                            saved=json.loads((job/'run_report.json').read_text())
                            self.assertIn('started_unix',saved['stages'][-1])
                            calls.append(args);clock.return_value+=2.5
                            if len(calls)==2 and failure is not None:raise failure
                        with contextlib.redirect_stdout(io.StringIO()):
                            try:
                                if backend=='fresh':
                                    with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=execute):
                                        fresh_pipeline.run(source,job,stages=['semantic','instances'])
                                else:
                                    config=dict(output=str(job),profile='test',sam_source='test')
                                    with patch('tools.lidar_annotator.cli.commands',return_value=[['dummy']]):
                                        cli.run(config,['semantic','instances'],runner=execute)
                            except (RuntimeError,KeyboardInterrupt):
                                if failure is None:raise
                    saved=json.loads((job/'run_report.json').read_text())
                    self.assertEqual(saved['status'],state)
                    self.assertEqual([r['elapsed_seconds'] for r in saved['stages']],[2.5,2.5])
                    self.assertTrue(all('finished_unix' in r for r in saved['stages']))
