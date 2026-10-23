"""Only exact, owned, idle task directories may be removed for regeneration."""
import fcntl,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .reset_results import clear_previous_results
from .test_lmdb_input import fixture
from .frame_history import record_test,test_history


def owner(path,source,frames):
    path.mkdir(parents=True)
    (path/'job.json').write_text(json.dumps(dict(schema='lidar_fresh_job_v1',source=str(source),frame_ids=frames)))
    (path/'result.npz').write_bytes(b'old derived result')
    return path


class ResetResultsTests(unittest.TestCase):
    def test_deletes_all_exact_runs_preserves_other_ranges_sources_and_originals(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'XDG_STATE_HOME':tmp}):
            root=Path(tmp);source=fixture(root);base={'output':str(root/'job')}
            old=owner(root/'job',source,[0]);run=owner(root/'job_runs/old',source,[0])
            other=owner(root/'job_runs/other',source,[1]);other_source=owner(root/'job_runs/source',root/'other-recording',[0])
            unknown=root/'job_runs/unknown';unknown.mkdir();(unknown/'keep').write_text('unowned')
            model=root/'model.pth';model.write_bytes(b'checkpoint')
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            for path in (old,run):record_test(source,[0],dict(output=str(path),review_output=str(path/'review'),input_format='lmdb'),'complete')
            removed=clear_previous_results(base,dict(output=str(run)),source,[0])
            self.assertEqual(set(removed),{old,run});self.assertFalse(old.exists());self.assertFalse(run.exists())
            self.assertTrue(other.is_dir());self.assertTrue(other_source.is_dir());self.assertTrue((unknown/'keep').is_file())
            self.assertEqual(model.read_bytes(),b'checkpoint')
            self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
            self.assertEqual(test_history(source),[])

    def test_regeneration_archives_published_lmdb_dataset_and_label_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);job=owner(root/'job',source,[0])
            published=job/'kitti/labels/image_2/000000.json';published.parent.mkdir(parents=True);published.write_text('reviewed label')
            points=job/'kitti/velodyne/000000.bin';points.parent.mkdir();points.write_bytes(b'raw points')
            version=job/'label_versions/old/000000.json';version.parent.mkdir(parents=True);version.write_text('previous version')
            base={'output':str(job)};clear_previous_results(base,base,source,[0])
            archives=root/'job_runs/label_archives'
            self.assertEqual([p.read_text() for p in archives.glob('*/kitti/labels/image_2/000000.json')],['reviewed label'])
            self.assertEqual([p.read_bytes() for p in archives.glob('*/kitti/velodyne/000000.bin')],[b'raw points'])
            self.assertEqual([p.read_text() for p in archives.glob('*/label_versions/old/000000.json')],['previous version'])

    def test_preview_and_invalid_selection_leave_existing_results_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);job=owner(root/'job',source,[0]);context={'output':str(job)}
            self.assertEqual(clear_previous_results(context,context,source,[0],dry_run=True),[job])
            self.assertTrue((job/'result.npz').is_file());self.assertFalse((job/'.run.lock').exists())
            with self.assertRaisesRegex(ValueError,'指定帧号'):clear_previous_results(context,context,source,[99])
            self.assertTrue((job/'result.npz').is_file())

    def test_another_running_job_blocks_all_deletions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);base=owner(root/'job',source,[0]);old=owner(root/'job_runs/old',source,[0])
            with (old/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'仍在运行'):clear_previous_results({'output':str(base)},{'output':str(old)},source,[0])
                self.assertTrue(base.is_dir());self.assertTrue(old.is_dir())

    def test_source_and_symlink_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);link=root/'link';link.symlink_to(source,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'符号链接'):clear_previous_results({'output':str(link)},{'output':str(link)},source,[0])
            with self.assertRaisesRegex(ValueError,'原始采集'):clear_previous_results({'output':str(source)},{'output':str(source)},source,[0])
            self.assertTrue((source/'index.lmdb/data.mdb').is_file())
