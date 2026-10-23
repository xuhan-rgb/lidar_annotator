"""Geometry and automatic startup export checks for the camera footprint."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from .camera_coverage import camera_coverage,save_coverage,generate_coverage
from .fresh_pipeline import run
from .test_lmdb_input import fixture


def calibration():
    # Camera looks along LiDAR +X; camera right is -Y and down is -Z.
    E=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],float)
    return np.array([[50,0,50],[0,50,50],[0,0,1]],float),E


class CameraCoverageTests(unittest.TestCase):
    def test_front_frustum_excludes_rear_and_image_outside_points(self):
        K,E=calibration();xyz=np.array([[5,0,0],[-5,0,0],[5,9,0],[5,0,9],[5,0,np.nan]],float)
        result=camera_coverage(K,E,(100,100),xyz,limits=(10,-1,1),resolution=1)
        np.testing.assert_array_equal(result['point_fov_mask'],[1,0,0,0,0])
        mask=result['bev_fov_mask'];self.assertEqual(mask.dtype,np.bool_);self.assertEqual(mask.shape,(20,20))
        self.assertTrue(mask[10,15]);self.assertFalse(mask[10,5]);self.assertFalse(mask[19,15])
        self.assertTrue(np.isfinite(result['boundary_xy']).all());self.assertEqual(result['boundary_xy'].shape[1:],(2,2))

    def test_vertical_frustum_intersects_height_slab(self):
        K,E=calibration();E[1,3]=10
        low=camera_coverage(K,E,(100,100),np.empty((0,3)),limits=(10,-1,1),resolution=1)
        high=camera_coverage(K,E,(100,100),np.empty((0,3)),limits=(10,9,11),resolution=1)
        self.assertFalse(low['bev_fov_mask'][10,15]);self.assertTrue(high['bev_fov_mask'][10,15])

    def test_saved_mask_has_coordinates_and_does_not_become_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            K,E=calibration();result=camera_coverage(K,E,(100,100),np.array([[5,0,0]]))
            save_coverage(Path(tmp),result,dict(frames=1,negative_supervision=False))
            with np.load(Path(tmp)/'visibility_mask.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['bev_fov_mask'],result['bev_fov_mask'])
                np.testing.assert_allclose(z['xy_min'],[-20,-20]);self.assertEqual(float(z['resolution']),.25)
                self.assertNotIn('negative_mask',z.files)
            self.assertNotIn('point_fov_mask',z.files)
            self.assertEqual([p.name for p in Path(tmp).iterdir()],['visibility_mask.npz'])

    def test_start_exports_masks_before_first_inference_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);job=root/'job'
            def execute(*args,**kwargs):
                self.assertTrue((job/'visibility_masks/visibility_mask.npz').is_file())
            with contextlib.redirect_stdout(io.StringIO()),patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=execute):run(source,job,stages=['semantic'])
            with np.load(job/'visibility_masks/visibility_mask.npz') as archive:report=json.loads(str(archive['metadata']))
            self.assertEqual(report['frames'],1);self.assertFalse(report['occlusion_checked']);self.assertFalse(report['negative_supervision'])

    def test_public_export_keeps_mask_next_to_portable_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);job=root/'job';public=root/'portable'
            with contextlib.redirect_stdout(io.StringIO()),patch('tools.lidar_annotator.fresh_pipeline.execute'):
                run(source,job,stages=['semantic'],dataset_output=public)
            with np.load(public/'labels/visibility_masks/visibility_mask.npz') as a,np.load(job/'visibility_masks/visibility_mask.npz') as b:
                np.testing.assert_array_equal(a['bev_fov_mask'],b['bev_fov_mask'])

    def test_contour_curves_cover_all_edges_without_breaking_dash_phase(self):
        from .camera_coverage import boundary_curves
        K,E=calibration();result=camera_coverage(K,E,(100,100),np.empty((0,3)))
        curves=boundary_curves(result['boundary_xy'])
        self.assertEqual(sum(len(curve)-1 for curve in curves),len(result['boundary_xy']))
        for curve in curves:np.testing.assert_array_equal(curve[0],curve[-1])

    def test_shared_mask_migrates_old_exports_and_ignores_point_count_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();K,E=calibration()
            image=raw/'image.png';Image.new('RGB',(100,100)).save(image)
            rows=[]
            for i in range(2):
                np.savez(raw/f'{i}.npz',cam_intrinsic=K,cam_extrinsic=E,xyz=np.zeros((i+1,3)))
                rows.append(dict(path=f'{i}.npz',image_path=str(image)))
            (raw/'index.json').write_text(json.dumps({'rows':rows}));(root/'job.json').write_text('{}')
            old=root/'visibility_masks';old.mkdir()
            (old/'index.json').write_text(json.dumps(dict(schema='camera_geometric_coverage_v1',rows=[dict(mask='000000.png',archive='000000.npz')])))
            (old/'000000.png').touch();(old/'000000.npz').touch()
            result=generate_coverage(root)
            self.assertEqual(result['frames'],2)
            self.assertEqual([p.name for p in old.iterdir()],['visibility_mask.npz'])
            # A different camera cannot silently reuse this common mask.
            np.savez(raw/'1.npz',cam_intrinsic=K*2,cam_extrinsic=E,xyz=np.zeros((2,3)))
            with self.assertRaisesRegex(ValueError,'标定或图像尺寸不同'):generate_coverage(root)
