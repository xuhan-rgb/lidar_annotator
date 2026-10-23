"""Starting LMDB annotation writes a portable public root, with separate work files."""
import fcntl,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .fresh_pipeline import run,stage
from .job_paths import work_directory
from .test_lmdb_input import fixture


class PublicOutputTests(unittest.TestCase):
    def setUp(self):
        import os
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        environment=patch.dict(os.environ,{'LIDAR_DATA_ROOT':temp.name});environment.start();self.addCleanup(environment.stop)

    def execute(self,*args,**kwargs):
        command=args[0];job=Path(command[command.index('--job')+1]);name=command[command.index('--stage')+1]
        stage(job,name,1)

    def test_start_writes_new_root_and_reuses_it_across_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public)
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=self.execute):run(source,work,stages=['render','catalog'],dataset_output=public)
            self.assertTrue((work/'raw/index.json').is_file());self.assertTrue((work/'review/index.html').is_file())
            self.assertTrue((work/'job.json').is_file());self.assertEqual(json.loads((work/'job.json').read_text())['dataset_output'],str(public))
            for name in ('image_2','velodyne','calib.txt','times.txt','poses.txt','metadata.npz'):self.assertTrue((public/name).exists(),name)
            for name in ('raw','review','job.json','logs'):self.assertFalse((public/name).exists(),name)
            label=public/'labels/image_2/000000.json';label.parent.mkdir(parents=True);label.write_text('reviewed manual annotation')
            second=work.parent/'new_run'
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=self.execute):run(source,second,stages=['render','catalog'],dataset_output=public)
            self.assertEqual(label.read_text(),'reviewed manual annotation')

    def test_resume_repairs_a_missing_image_even_when_metadata_still_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public)
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=self.execute):run(source,work,stages=['render'],dataset_output=public)
            image=public/'image_2/000000.png';original=image.read_bytes();image.unlink()
            self.assertTrue((public/'metadata.npz').is_file())
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=self.execute):run(source,work,stages=['render'],dataset_output=public)
            self.assertEqual(image.read_bytes(),original)

    def test_nonstandard_old_work_root_is_rejected_without_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';public.mkdir()
            (public/'job.json').write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=str(source),frame_ids=[0])))
            (public/'raw').mkdir();(public/'raw/old.npz').write_bytes(b'keep original cache')
            work=work_directory(source,public)
            with self.assertRaises(ValueError):run(source,work,stages=['render'],dataset_output=public)
            self.assertEqual((public/'raw/old.npz').read_bytes(),b'keep original cache')
            self.assertFalse(list(work.parent.glob('*_legacy_*')))

    def test_cancelled_conversion_resumes_without_rereading_lmdb(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public);raw=work/'raw'
            read_lmdb(source,raw,kitti_output=False)
            (raw/'status.json').write_text(json.dumps(dict(status='cancelled',phase='kitti')))
            original=(raw/'frames/000000.npz').read_bytes()
            with patch('tools.lidar_annotator.fresh_pipeline.read_raw',side_effect=AssertionError('must reuse complete import')),patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=self.execute):
                run(source,work,stages=['render'],dataset_output=public)
            self.assertEqual((work/'raw/frames/000000.npz').read_bytes(),original)
            index=json.loads((work/'raw/index.json').read_text());self.assertTrue(Path(index['rows'][0]['image_path']).is_file())
            self.assertTrue((public/'velodyne/000000.bin').is_file());self.assertFalse((public/'raw').exists())

    def test_failed_resumed_conversion_records_error(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public);raw=work/'raw';read_lmdb(source,raw,kitti_output=False)
            with patch('tools.lidar_annotator.kitti_input.export_imported_kitti',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):run(source,work,stages=['render'],dataset_output=public)
            status=json.loads((work/'raw/status.json').read_text());self.assertEqual(status['status'],'failed');self.assertEqual(status['kitti_export_error'],'disk full')

    def test_active_current_task_cannot_be_started_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public);work.mkdir(parents=True)
            owner=work/'job.json';owner.write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=str(source),frame_ids=[0],source_readonly=True,dataset_output=str(public))))
            with (work/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'already running'):run(source,work,stages=['render'],dataset_output=public)
                self.assertTrue(owner.exists())

    def test_dry_run_creates_neither_work_nor_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);public=root/'dataset';work=work_directory(source,public)
            run(source,work,dry_run=True,dataset_output=public)
            self.assertFalse(public.exists());self.assertFalse(work.exists())

if __name__=='__main__':unittest.main()
