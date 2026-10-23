import hashlib,io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from .test_lmdb_input import fixture
from .kitti_input import camera_channels,export_imported_kitti,export_lmdb_kitti,frame_records,read_kitti,source_image


class KittiInputTests(unittest.TestCase):

    def test_raw_directory_can_move_behind_its_original_symlink(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';read_lmdb(fixture(root),raw)
            relocated=root/'external/raw';relocated.parent.mkdir();raw.rename(relocated)
            raw.symlink_to(relocated,target_is_directory=True)
            self.assertTrue(export_imported_kitti(raw,raw/'kitti')['reused'])

    def test_same_source_missing_files_are_repaired_without_overwriting_labels(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';read_lmdb(fixture(root),raw);out=raw/'kitti'
            points=out/'velodyne/000000.bin';before=points.read_bytes();points.unlink();(out/'metadata.npz').unlink()
            label=out/'labels/image_2/000000.json';label.parent.mkdir(parents=True);label.write_text('reviewed annotation')
            image=out/'image_2/000000.png';image_before=image.read_bytes()
            report=export_imported_kitti(raw,out)
            self.assertEqual(points.read_bytes(),before);self.assertTrue((out/'metadata.npz').is_file())
            self.assertEqual(image.read_bytes(),image_before);self.assertEqual(label.read_text(),'reviewed annotation')
            self.assertEqual(report['repaired_files'],2)

    def test_marker_only_output_can_be_restored_but_modified_raw_files_are_protected(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';read_lmdb(fixture(root),raw);out=raw/'kitti'
            for child in list(out.iterdir()):
                if child.name=='.lidar_annotator_kitti.json':continue
                if child.is_dir():shutil.rmtree(child)
                else:child.unlink()
            report=export_imported_kitti(raw,out);self.assertGreater(report['repaired_files'],0)
            image=out/'image_2/000000.png';image.write_bytes(b'manually modified image');(out/'metadata.npz').unlink()
            with self.assertRaisesRegex(ValueError,'已修改'):export_imported_kitti(raw,out)
            self.assertEqual(image.read_bytes(),b'manually modified image')

    def test_interruption_after_a_repaired_file_is_published_can_resume(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';read_lmdb(fixture(root),raw);out=raw/'kitti';(out/'metadata.npz').unlink()
            other=root/'other_raw';shutil.copytree(raw,other,ignore=shutil.ignore_patterns('kitti'))
            index=json.loads((other/'index.json').read_text())
            for row in index['rows']:row['image_path']=str(other/'images'/Path(row['image_path']).name)
            (other/'index.json').write_text(json.dumps(index))
            original_replace=Path.replace
            def interrupted(path,target):
                result=original_replace(path,target)
                if Path(target)==out/'metadata.npz':raise KeyboardInterrupt
                return result
            with patch.object(Path,'replace',interrupted):
                with self.assertRaises(KeyboardInterrupt):export_imported_kitti(other,out)
            self.assertTrue(export_imported_kitti(other,out)['reused'])

    def test_repeated_raw_export_preserves_published_labels(self):
        from .lmdb_input import read_lmdb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);raw=root/'raw';read_lmdb(source,raw)
            label=raw/'kitti/labels/image_2/000000.json';label.parent.mkdir(parents=True);label.write_text('reviewed annotation')
            report=export_imported_kitti(raw,raw/'kitti')
            self.assertTrue(report['reused']);self.assertEqual(label.read_text(),'reviewed annotation')


    def test_lmdb_export_import_roundtrip_and_portability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            kitti=root/'portable';report=export_lmdb_kitti(source,kitti,count=10)
            self.assertEqual(report['exported_frames'],1)
            with np.load(kitti/'metadata.npz') as metadata, Image.open(kitti/'image_2/000000.png') as png, Image.open(io.BytesIO(metadata['jpeg_0'].tobytes())) as jpg:
                np.testing.assert_array_equal(np.asarray(png.convert('RGB')),np.asarray(jpg.convert('RGB')))
            values=np.fromfile(kitti/'velodyne/000000.bin',dtype='<f4').reshape(-1,4)
            np.testing.assert_array_equal(values[:,[0,1,2,3]],[[0,0,5,10],[0,0,-1,10]])
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
            shutil.rmtree(source)
            raw=read_kitti(kitti,root/'raw')
            self.assertEqual(raw['input_format'],'kitti');self.assertFalse(raw['labels_generated'])
            with np.load(root/'raw/frames/000000.npz') as a:
                self.assertEqual(a['lidar_points'].shape,(2,5));self.assertEqual(int(a['lidar_timestamp_ns']),1000)
                np.testing.assert_allclose(a['cam_intrinsic'],[[100,0,32],[0,100,24],[0,0,1]])
                np.testing.assert_allclose(a['pose_world'],np.zeros(4))
            np.testing.assert_array_equal(np.asarray(source_image(kitti,0)),np.asarray(Image.open(kitti/'image_2/000000.png').convert('RGB')))
            self.assertFalse(any((root/'raw').glob('**/*label*')))

    def test_standalone_folder_without_metadata_imports_kitti_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'image_2').mkdir(parents=True);(seq/'velodyne').mkdir()
            Image.new('RGB',(4,3),'red').save(seq/'image_2/000000.png')
            np.array([[1,2,3,4]],dtype='<f4').tofile(seq/'velodyne/000000.bin')
            (seq/'calib.txt').write_text('P2: 10 0 2 0 0 10 1 0 0 0 1 0\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            (seq/'poses.txt').write_text('1 0 0 0 0 1 0 0 0 0 1 0\n');(seq/'times.txt').write_text('0.000000000\n')
            read_kitti(seq,root/'raw',preview=True)
            self.assertTrue((root/'raw/review/visualization/000000.png').is_file())
            with np.load(root/'raw/frames/000000.npz') as a:
                np.testing.assert_array_equal(a['lidar_points'],[[1,2,3,4]])
                np.testing.assert_allclose(a['pose_world'],[0,0,0,0])

    def test_missing_frame_and_nonempty_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);kitti=root/'kitti';export_lmdb_kitti(source,kitti)
            with self.assertRaises(ValueError):read_kitti(kitti,root/'raw',frames=[9])
            out=root/'occupied';out.mkdir();(out/'keep').write_text('x')
            with self.assertRaises(ValueError):read_kitti(kitti,out)

    def test_camera_selection_uses_matching_projection_and_keeps_pose_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'velodyne').mkdir(parents=True)
            for cam,color in [('image_0','red'),('image_2','blue'),('image_3','green')]:
                (seq/cam).mkdir();Image.new('RGB',(20,10),color).save(seq/cam/'000000.png');Image.new('RGB',(20,10),color).save(seq/cam/'000001.png')
            np.array([[1,2,10,1]],dtype='<f4').tofile(seq/'velodyne/000000.bin');np.array([[1,2,10,1]],dtype='<f4').tofile(seq/'velodyne/000001.bin')
            Tr='1 0 0 2 0 1 0 3 0 0 1 0';R='0 -1 0 1 0 0 0 0 1'
            P0='10 0 5 20 0 10 4 0 0 0 1 0';P2='10 0 5 0 0 10 4 0 0 0 1 0'
            P3='10 0 5 0 0 10 4 0 0 0 1 0';Tr3='0 0 1 0 0 1 0 0 -1 0 0 0'
            (seq/'calib.txt').write_text(f'P0: {P0}\nP2: {P2}\nP3: {P3}\nR0_rect: {R}\nTr_velo_to_cam: {Tr}\nTr_velo_to_cam_3: {Tr3}\n')
            (seq/'times.txt').write_text('0\n0.1\n')
            rect=np.eye(4);rect[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]]);tr=np.eye(4);tr[:3,:3]=np.eye(3);tr[:3,3]=[2,3,0];rectE=rect@tr
            lidar_poses=[]
            for x in [1,2]:
                t=np.eye(4);t[0,3]=x;lidar_poses.append(t)
            (seq/'poses.txt').write_text(''.join(' '.join(map(str,(rectE@t@np.linalg.inv(rectE))[:3].ravel()))+'\n' for t in lidar_poses))
            (seq/'image_masks').mkdir();Image.new('RGB',(20,10),'black').save(seq/'image_masks/000000.png')
            self.assertEqual(camera_channels(seq),['image_0','image_2','image_3'])
            self.assertEqual(frame_records(seq)[1][0]['camera'],'image_2')
            self.assertEqual(source_image(seq,0,'image_0').getpixel((0,0)),(255,0,0))
            read_kitti(seq,root/'raw0',camera='image_0');read_kitti(seq,root/'raw2',camera='image_2');read_kitti(seq,root/'raw3',camera='image_3')
            with np.load(root/'raw0/frames/000000.npz') as a0,np.load(root/'raw2/frames/000000.npz') as a2,np.load(root/'raw3/frames/000000.npz') as a3:
                self.assertAlmostEqual(float(a0['pixels_uv'][0,0]),2.0);self.assertAlmostEqual(float(a2['pixels_uv'][0,0]),0.0);self.assertNotEqual(float(a2['pixels_uv'][0,0]),float(a3['pixels_uv'][0,0]))
            with np.load(root/'raw0/odometry.npz') as a0,np.load(root/'raw2/odometry.npz') as a2,np.load(root/'raw3/odometry.npz') as a3:
                np.testing.assert_allclose(a0['transforms'],a2['transforms']);np.testing.assert_allclose(a0['transforms'],a3['transforms']);np.testing.assert_allclose(a0['transforms'][1,:3,3],[1,0,0])
            self.assertEqual(json.loads((root/'raw0/index.json').read_text())['rows'][0]['source_index_record']['camera'],'image_0')

    def test_imported_raw_helper_and_missing_camera_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);from .lmdb_input import read_lmdb
            raw=root/'raw';read_lmdb(source,raw);portable=root/'portable';export_imported_kitti(raw,portable)
            self.assertTrue(export_imported_kitti(raw,portable)['reused'])
            self.assertTrue((portable/'metadata.npz').is_file())
            (portable/'image_0').mkdir();(portable/'image_0/000000.png').write_bytes((portable/'image_2/000000.png').read_bytes())
            (portable/'calib.txt').write_text('P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            with self.assertRaisesRegex(ValueError,'P0'):read_kitti(portable,root/'bad',camera='image_0')
            with self.assertRaises(ValueError):export_imported_kitti(raw,portable)

    def test_sparse_local_frame_ids_use_matching_times_and_poses_and_per_frame_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'image_2').mkdir(parents=True);(seq/'velodyne').mkdir();(seq/'calib').mkdir()
            for i in (2,5):
                Image.new('RGB',(4,3),(i,0,0)).save(seq/'image_2'/f'{i:06d}.png');np.array([[1,2,3,4]],dtype='<f4').tofile(seq/'velodyne'/f'{i:06d}.bin')
                (seq/'calib'/f'{i:06d}.txt').write_text('P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            (seq/'times.txt').write_text('0\n0.1\n0.200000001\n0.3\n0.4\n0.5\n')
            (seq/'poses.txt').write_text(''.join(f'1 0 0 {i} 0 1 0 0 0 0 1 0\n' for i in range(6)))
            report=read_kitti(seq,root/'raw');self.assertEqual(report['frame_ids'],[2,5])
            with np.load(root/'raw/frames/000002.npz') as a2,np.load(root/'raw/frames/000005.npz') as a5:
                self.assertEqual(int(a2['lidar_timestamp_ns']),200000001);self.assertEqual(int(a5['lidar_timestamp_ns']),500000000)
            with np.load(root/'raw/odometry.npz') as odometry:
                np.testing.assert_allclose(odometry['transforms'][:,0,3],[2,5])

    def test_pose_uses_fixed_root_reference_and_ignores_cached_pose_for_odometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'image_2').mkdir(parents=True);(seq/'velodyne').mkdir();(seq/'calib').mkdir()
            for i in range(2):
                Image.new('RGB',(4,3),'red').save(seq/'image_2'/f'{i:06d}.png');np.array([[1,2,3,4]],dtype='<f4').tofile(seq/'velodyne'/f'{i:06d}.bin')
            calibration='P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n'
            (seq/'calib.txt').write_text(calibration)
            changed='P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr_velo_to_cam: 0 0 1 0 0 1 0 0 -1 0 0 0\n'
            (seq/'calib/000001.txt').write_text(changed)
            (seq/'times.txt').write_text('0\n0.1\n')
            identity=np.eye(4);translated=np.eye(4);translated[0,3]=2
            (seq/'poses.txt').write_text(''.join(' '.join(map(str,t[:3].ravel()))+'\n' for t in (identity,translated)))
            pose_hash=hashlib.sha256((seq/'poses.txt').read_bytes()).hexdigest()
            np.savez(seq/'metadata.npz',global_frame_ids=np.array([0,1]),local_frame_ids=np.array([0,1]),source_pose_world=np.array([[99,0,0,0],[99,0,0,0]]),poses_txt_sha256=np.array(pose_hash))
            read_kitti(seq,root/'raw')
            with np.load(root/'raw/odometry.npz') as odometry:np.testing.assert_allclose(odometry['transforms'][:,0,3],[0,2])
            with np.load(root/'raw/frames/000001.npz') as frame:np.testing.assert_allclose(frame['pose_world'],[2,0,0,0])

    def test_missing_times_keeps_projection_but_marks_temporal_data_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'image_2').mkdir(parents=True);(seq/'velodyne').mkdir()
            Image.new('RGB',(4,3),'red').save(seq/'image_2/000000.png');np.array([[1,2,3,4]],dtype='<f4').tofile(seq/'velodyne/000000.bin')
            (seq/'calib.txt').write_text('P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            report=read_kitti(seq,root/'raw');self.assertFalse(report['timestamps_available'])
            with np.load(root/'raw/frames/000000.npz') as frame:
                self.assertTrue(np.isnan(frame['lidar_timestamp_ns']))
                self.assertTrue(np.isfinite(frame['pixels_uv']).any())
            with np.load(root/'raw/odometry.npz') as odometry:self.assertTrue(np.isnan(odometry['timestamps']).all())

    def test_lmdb_export_without_pose_does_not_invent_pose_file(self):
        import lmdb
        from .lmdb_input import decode_frame,msgpack
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);shard=source/'shards/shard_000000.lmdb'
            with lmdb.open(str(shard),map_size=4*1024*1024) as env:
                with env.begin() as txn:payload=txn.get(b'frame/0')
                with np.load(io.BytesIO(payload),allow_pickle=False) as z:arrays={k:z[k] for k in z.files if k!='pose_world'}
                updated=io.BytesIO();np.savez_compressed(updated,**arrays);payload=updated.getvalue()
                with env.begin(write=True) as txn:txn.put(b'frame/0',payload)
            from .lmdb_input import frame_records as lmdb_records
            _,records=lmdb_records(source);records[0]['payload_sha256']=hashlib.sha256(payload).hexdigest()
            with lmdb.open(str(source/'index.lmdb'),map_size=4*1024*1024) as env:
                with env.begin(write=True) as txn:txn.put(b'frame_index/000000000000',msgpack.packb(records[0],use_bin_type=True))
            kitti=root/'kitti';export_lmdb_kitti(source,kitti)
            self.assertFalse((kitti/'poses.txt').exists())
            read_kitti(kitti,root/'raw')
            with np.load(root/'raw/odometry.npz') as odometry:self.assertTrue(np.isnan(odometry['transforms']).all())

    def test_missing_poses_remain_unavailable_and_input_changes_invalidate_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seq=root/'seq';(seq/'image_2').mkdir(parents=True);(seq/'velodyne').mkdir()
            Image.new('RGB',(4,3),'red').save(seq/'image_2/000000.png');np.array([[1,2,3,4]],dtype='<f4').tofile(seq/'velodyne/000000.bin')
            (seq/'calib.txt').write_text('P2: 1 0 0 0 0 1 0 0 0 0 1 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n');(seq/'times.txt').write_text('0\n')
            read_kitti(seq,root/'raw0');
            with np.load(root/'raw0/frames/000000.npz') as raw: self.assertTrue(np.isnan(raw['pose_world']).all())
            with np.load(root/'raw0/odometry.npz') as odo:self.assertTrue(np.isnan(odo['transforms']).all())
            def digest(out):
                read_kitti(seq,out);return json.loads((out/'index.json').read_text())['rows'][0]['original_payload_sha256']
            base=digest(root/'hash0')
            (seq/'velodyne/000000.bin').write_bytes(np.array([[2,2,3,4]],dtype='<f4').tobytes());self.assertNotEqual(base,digest(root/'hash_points'))
            Image.new('RGB',(4,3),'blue').save(seq/'image_2/000000.png');self.assertNotEqual(base,digest(root/'hash_image'))
            (seq/'calib.txt').write_text((seq/'calib.txt').read_text().replace('P2: 1','P2: 2'));self.assertNotEqual(base,digest(root/'hash_calib'))
            np.savez(seq/'metadata.npz',changed=np.array([1]));self.assertNotEqual(base,digest(root/'hash_meta'))


if __name__=='__main__':unittest.main()
