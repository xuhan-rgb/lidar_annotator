"""Final global person positions must agree in frames, maps and portable labels."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .test_optimized_map import fixture
from .optimized_map import build_optimized_map,load_optimized_map
from .live_preview import load_frame
from .fresh_pipeline import catalog
from .dataset_labels import _lidar_bundle
from .recipe_config import default_recipe
from tools.autolabel.zgc_labeling.finalize_person_positions import run,load_positions


def final_fixture(root):
    context,xyz,poses=fixture(root);video=root/'temporal/map_review/sam3_full_video'
    tracks=video/'interior_half_review/consistent_tracks/centers_only.npz'
    with np.load(tracks) as z:data={k:z[k] for k in z.files}
    data.update(reference_xy_map=np.array([[.2,.15],[.2,.15]]),position_source=np.array(['observed','linear_interpolation_only']))
    np.savez_compressed(tracks,**data)
    visual=root/'data/visual_instances';visual.mkdir(parents=True)
    people=root/'data/person/metadata';people.mkdir(parents=True)
    for frame in range(2):
        (visual/f'{frame:06d}.json').write_text(json.dumps({'records':[dict(prompt='person',track_id=7,bbox_xyxy=[3,3,7,6])]}))
        (people/f'{frame:06d}.json').write_text(json.dumps({'instances':[dict(sam_source_instance_id=7,
            person_class_confirmed=True,status='accepted_xy_from_all_exclusive_mask_points',
            yolo_match=dict(score=.9,bbox_xyxy=[3,3,7,6]),exclusive_support_points=[0,1])]}))
    (video/'native_to_display_ids.json').write_text(json.dumps({'7':7}))
    source=root/'data/labels/zgc/index.json';index=json.loads(source.read_text())
    for row in index['rows']:row['frame_id']=row['original_frame_id']
    source.write_text(json.dumps(index))
    config=dict(source_index=str(source),pose_npz=str(root/'temporal/poses.npz'),temporal_output=str(root/'temporal'),
                person_root=str(root/'data/person'),vehicle_root=str(root/'data/vehicle'),coordinate_assumption='synthetic')
    path=root/'backend.json';path.write_text(json.dumps(config));(root/'recipe.json').write_text(json.dumps(default_recipe()))
    run(path)
    return context,xyz,poses,tracks,path


class FinalPersonPositionsTests(unittest.TestCase):
    def test_map_distinguishes_supported_positions_from_track_only(self):
        from PySide6.QtWidgets import QApplication
        from .map_view import MapCanvas
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            context,_,_,_,_=final_fixture(Path(directory));data=build_optimized_map(context,0)
            data['person_track_only']=np.array([False,True])
            canvas=MapCanvas();canvas.set_map(data)
            canvas.show_frames=False;canvas.show_start=False;canvas.show_cars=False;canvas.show_person_paths=False
            try:
                with patch('tools.lidar_annotator.map_view.QPainter') as painter:canvas.paintEvent(None)
                self.assertEqual(painter.return_value.drawEllipse.call_count,1)
            finally:canvas.close()

    def test_final_file_controls_frame_map_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);context,xyz,poses,tracks,_=final_fixture(root)
            before=hashlib.sha256(tracks.read_bytes()).hexdigest()
            final=load_positions(tracks);data=build_optimized_map(context,0)
            np.testing.assert_allclose(data['person_xyz'][:,:2],[[.2,.15],[.2,.15]],atol=1e-6)
            np.testing.assert_array_equal(data['person_track_only'],final['track_only'])
            with patch('tools.lidar_annotator.dataset_labels.export_labels'):catalog(root)
            for local,frame in enumerate((63,64)):
                view=load_frame(context,frame,mode='combined')
                expected=final['corrected_point_lidar'][local]
                np.testing.assert_allclose(view['centres'],expected[None,:2])
                np.testing.assert_array_equal(view['centre_track_only'],final['track_only'][[local]])
                path=root/'labels'/f'{frame:06d}.npz'
                with np.load(path) as z:
                    np.testing.assert_allclose(z['person_xyz'],expected[None])
                    np.testing.assert_array_equal(z['person_track_only'],final['track_only'][[local]])
                    np.testing.assert_array_equal(z['xyz'],xyz)
                binary=root/f'{frame}.bin';np.c_[xyz,np.zeros(len(xyz))].astype('<f4').tofile(binary)
                content,_=_lidar_bundle({'label_path':path},'image_2',local,frame,binary)
                np.testing.assert_allclose(content['persons'][0]['position_xyz'],expected)
                self.assertEqual(content['persons'][0]['track_only'],bool(final['track_only'][local]))
                self.assertEqual(content['persons'][0]['position_policy'],'global_reference_xy_v1')
            self.assertEqual(hashlib.sha256(tracks.read_bytes()).hexdigest(),before)

    def test_new_final_positions_invalidate_map_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);context,_,_,tracks,config=final_fixture(root)
            first=load_optimized_map(context,0)
            with np.load(tracks) as z:data={k:z[k] for k in z.files}
            data['reference_xy_map']+=.1;np.savez_compressed(tracks,**data)
            run(config);updated=load_optimized_map(context,0)
            self.assertFalse(updated['cache_reused'])
            np.testing.assert_allclose(updated['person_xyz'][:,:2],first['person_xyz'][:,:2]+.1,atol=1e-6)
