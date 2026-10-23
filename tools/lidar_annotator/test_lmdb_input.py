import hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import lmdb,numpy as np
from PIL import Image
from .lmdb_input import msgpack,read_lmdb


def fixture(root,corrupt=False):
    source=root/'recording';(source/'shards').mkdir(parents=True)
    image=io.BytesIO();Image.new('RGB',(64,48),'gray').save(image,format='JPEG')
    arrays=dict(lidar_points=np.array([[0,0,5,10,0],[0,0,-1,10,.1]],np.float32),cam_intrinsic=np.array([[100,0,32],[0,100,24],[0,0,1]],float),cam_extrinsic=np.eye(4),image_jpeg_bytes=np.frombuffer(image.getvalue(),np.uint8),lidar_timestamp_ns=np.array(1000,np.int64),pose_world=np.zeros(4),frame_id=np.array(0))
    out=io.BytesIO();np.savez_compressed(out,**arrays);payload=out.getvalue()
    with lmdb.open(str(source/'shards/shard_000000.lmdb'),map_size=4*1024*1024) as env:
        with env.begin(write=True) as txn:txn.put(b'frame/0',payload)
    with lmdb.open(str(source/'index.lmdb'),map_size=4*1024*1024) as env:
        with env.begin(write=True) as txn:txn.put(b'frame_index/000000000000',msgpack.packb(dict(global_frame_id=0,shard_id=0,lmdb_key='frame/0',timestamp_ns=1000,payload_sha256='0'*64 if corrupt else hashlib.sha256(payload).hexdigest()),use_bin_type=True))
    return source


class LmdbInputTests(unittest.TestCase):
    def test_projection_points_are_colored_by_camera_depth(self):
        from .lmdb_input import preview_frame
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'projection.png'
            arrays=dict(xyz=np.array([[-15,0,5],[15,0,5]],float),pixels_uv=np.array([[10,10],[30,10]],float),in_image=np.array([True,True]),camera_depth=np.array([5,35],float))
            preview_frame(Image.new('RGB',(64,48),'gray'),arrays,path,1)
            with Image.open(path) as image:
                self.assertNotEqual(image.getpixel((30,100)),image.getpixel((50,100)))
                self.assertNotEqual(image.getpixel((30,100)),(0,220,220))
                near=image.getpixel((30,100));far=image.getpixel((50,100))
                self.assertGreater(near[0],near[2]);self.assertGreater(far[2],far[0])

    def test_cancelled_read_records_cancelled_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root)
            with patch('tools.lidar_annotator.lmdb_input.decode_frame',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):read_lmdb(source,root/'output')
            self.assertEqual(json.loads((root/'output/status.json').read_text())['status'],'cancelled')

    def test_cancelled_kitti_conversion_records_cancelled_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root)
            with patch('tools.lidar_annotator.kitti_input.export_imported_kitti',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):read_lmdb(source,root/'output')
            status=json.loads((root/'output/status.json').read_text());self.assertEqual(status['status'],'cancelled');self.assertEqual(status['phase'],'kitti')

    def test_read_export_projection_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.mdb')}
            report=read_lmdb(source,root/'output',preview=True)
            self.assertEqual(report['imported_frames'],1);self.assertFalse(report['labels_generated'])
            with np.load(root/'output/frames/000000.npz') as a:
                np.testing.assert_allclose(a['pixels_uv'][0],[32,24]);np.testing.assert_array_equal(a['in_image'],[True,False]);np.testing.assert_array_equal(a['xyz'],a['lidar_points'][:,:3])
            self.assertTrue((root/'output/review/visualization/000000.png').is_file())
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.mdb')})

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);read_lmdb(source,root/'output',dry_run=True);self.assertFalse((root/'output').exists())

    def test_payload_hash_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root,corrupt=True)
            with self.assertRaises(ValueError):read_lmdb(source,root/'output')
            self.assertEqual(json.loads((root/'output/status.json').read_text())['status'],'failed')

    def test_unknown_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root)
            with self.assertRaises(ValueError):read_lmdb(source,root/'output',frames=[99])


if __name__=='__main__':unittest.main()
