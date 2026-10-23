import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .test_lmdb_input import fixture
from .fresh_pipeline import run,prepare,poses,stage
from .cli import STAGES


class FreshPipelineTests(unittest.TestCase):
    def test_default_plan_generates_all_stages_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);job=root/'job'
            result=run(source,job,dry_run=True)
            self.assertEqual([r['stage'] for r in result['plan']],list(STAGES))
            self.assertFalse(job.exists())
            self.assertEqual(result['job']['frame_ids'],[0])

    def test_subset_identity_cannot_reuse_another_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);job=root/'job';job.mkdir()
            (job/'job.json').write_text(json.dumps({'source':'different'}))
            with self.assertRaisesRegex(ValueError,'different source'):run(source,job,stages=['render'])

    def test_pose_segments_rebase_and_reject_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);(job/'raw').mkdir();prepare(job)
            np.savez(job/'raw/odometry.npz',poses=[[10,20,0,0],[10.1,20,0,0],[10.2,20,0,0]],timestamps=np.array([0,100000000,3000000000],np.int64))
            poses(job)
            with np.load(job/'temporal/poses.npz') as z:
                np.testing.assert_allclose(z['transforms'][0],np.eye(4));np.testing.assert_array_equal(z['segments'],[0,0,1])
            np.savez(job/'raw/odometry.npz',poses=[[np.nan,0,0,0]],timestamps=[0])
            with self.assertRaisesRegex(ValueError,'finite'):poses(job)

    def test_failure_records_failed_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);job=root/'job'
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=RuntimeError('inference failed')):
                with self.assertRaises(RuntimeError):run(source,job,stages=['semantic'])
            report=json.loads((job/'run_report.json').read_text())
            self.assertEqual(report['status'],'failed');self.assertEqual(report['stages'][0]['status'],'failed')
            self.assertFalse(report['labels_promoted'])

    def test_real_raw_only_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);job=root/'job'
            output=io.StringIO()
            with contextlib.redirect_stdout(output),patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=lambda *a,**k:stage(job,'render',1)):
                result=run(source,job,stages=['render'])
            self.assertIn(f'可视化目录：{job}/review/visualization',output.getvalue())
            self.assertLess(output.getvalue().index('可视化目录：'),output.getvalue().index('Starting render'))
            self.assertEqual(result['status'],'complete')
            self.assertTrue((job/'review/visualization/000000.png').is_file())

    def test_partial_debug_exports_without_temporal_outputs(self):
        from .fresh_pipeline import catalog
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);job=root/'job'
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=lambda *a,**k:stage(job,'render',1)):
                run(source,job,stages=['render'])
            catalog(job)
            result=json.loads((job/'annotations.json').read_text())
            self.assertFalse(result['optimized_positions_available'])
            self.assertEqual(result['frames'],1)
            with np.load(job/'labels/000000.npz') as labels:
                self.assertIn('xyz',labels.files)
                self.assertNotIn('person_xyz',labels.files)
