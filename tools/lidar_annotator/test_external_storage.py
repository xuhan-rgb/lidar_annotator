"""Storage roots are external configuration, without legacy source-tree aliases."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .storage import data_root,directory,sample_path,default_output,default_source
from .job_paths import work_directory


class ExternalStorageTests(unittest.TestCase):
    def environment(self,root):
        return patch.dict(os.environ,{'XDG_CONFIG_HOME':str(root/'config'),
                                    'XDG_DATA_HOME':str(root/'share'),'LIDAR_DATA_ROOT':''})

    def test_xdg_default_is_outside_the_source_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.environment(root):
                self.assertEqual(data_root(),root/'share/lidar_annotator')
                self.assertEqual(sample_path('testdata/lmdb_sample'),data_root()/'samples/testdata/lmdb_sample')

    def test_user_configuration_controls_all_data_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=root/'config/lidar_annotator/storage.json';cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({'data_root':str(root/'data'),'lmdb_source':str(root/'recording')}))
            with self.environment(root):
                self.assertEqual(directory('datasets'),root/'data/datasets')
                self.assertEqual(directory('jobs'),root/'data/jobs')
                self.assertEqual(default_source('lmdb'),root/'recording')
                self.assertEqual(default_output('lmdb'),root/'data/datasets/lmdb_import')
                job=work_directory(root/'recording',default_output('lmdb'))
                self.assertEqual(job.parent,root/'data/jobs')
                self.assertEqual(job,work_directory(root/'recording',default_output('lmdb')))
                self.assertNotEqual(job,work_directory(root/'other',default_output('lmdb')))

    def test_environment_overrides_the_configured_data_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.environment(root),patch.dict(os.environ,{'LIDAR_DATA_ROOT':str(root/'override')}):
                self.assertEqual(default_output('kitti'),root/'override/jobs/kitti_import')

    def test_annotation_rejects_work_and_data_inside_source_directory(self):
        from .fresh_pipeline import run
        from .test_lmdb_input import fixture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=fixture(root);code=root/'code';code.mkdir()
            with patch('tools.lidar_annotator.fresh_pipeline.REPO',code):
                with self.assertRaisesRegex(ValueError,'工作目录必须位于源码目录之外'):
                    run(source,code/'outputs',dry_run=True)
                with self.assertRaisesRegex(ValueError,'标准数据目录必须位于源码目录之外'):
                    run(source,root/'job',dataset_output=code/'dataset',dry_run=True)


if __name__=='__main__':unittest.main()
