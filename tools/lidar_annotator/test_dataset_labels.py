"""Portable label schema, original row alignment and preservation of user changes."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from .dataset_labels import export_labels
from .fresh_pipeline import run,stage,catalog
from .kitti_input import export_lmdb_kitti
from .test_lmdb_input import fixture


def job_fixture(root,kitti=True):
    source=fixture(root);job=root/'job'
    if kitti:
        sequence=root/'sequence';export_lmdb_kitti(source,sequence)
        with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=lambda *a,**k:stage(job,'render',1)):run(sequence,job,stages=['render'],camera='image_2')
    else:
        with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=lambda *a,**k:stage(job,'render',1)):run(source,job,stages=['render'])
        sequence=job/'kitti'
    mask=job/'masks/000000.png';mask.parent.mkdir();Image.fromarray(np.array([[0,1],[2,255]],np.uint8)).save(mask)
    conf=mask.with_suffix('.npz');np.savez_compressed(conf,confidence=np.array([[.6,.7],[.8,0]],np.float16))
    instances=job/'data/visual_instances/000000.npz';instances.parent.mkdir(parents=True)
    masks=np.array([[[0,1],[0,0]]],bool)
    np.savez_compressed(instances,masks_packed=np.packbits(masks.reshape(1,-1),axis=1,bitorder='little'),mask_shape=np.array([2,2]))
    instances.with_suffix('.json').write_text(json.dumps({'records':[dict(prompt='person',score=.8,bbox_xyxy=[1,0,2,1],source_instance_id=10,track_id=10)]}))
    people=job/'data/person/metadata/000000.json';people.parent.mkdir(parents=True);people.write_text(json.dumps({'instances':[dict(sam_source_instance_id=10,status='accepted_xy_from_all_exclusive_mask_points')]}))
    ids=job/'temporal/map_review/sam3_full_video/native_to_display_ids.json';ids.parent.mkdir(parents=True);ids.write_text(json.dumps({'10':1}))
    yolo=job/'data/yolo/metadata/000000.json';yolo.parent.mkdir(parents=True);yolo.write_text(json.dumps({'detections':[dict(name='person',class_id=0,score=.9,bbox_xyxy=[1,0,2,1])]}))
    with np.load(job/'raw/frames/000000.npz') as z:arrays={k:z[k] for k in z.files}
    arrays.update(point_raw_labels=np.array([7,9],np.uint8),training_roi_mask=np.array([True,False]),person_xyz=np.array([[1.,2.,3.]]),person_track_ids=np.array([1]),person_position_source=np.array(['observed']),vehicle_box_3d=np.array([[5.,6.,1.,4.,2.,1.5,.2]]),vehicle_track_ids=np.array([3]))
    label=job/'labels/000000.npz';label.parent.mkdir();np.savez_compressed(label,**arrays)
    row=dict(frame_id=0,local_frame=0,label='labels/000000.npz',semantic_mask=str(mask),semantic_confidence=str(conf),visual_instances=str(instances),person_metadata=str(people),vehicle_metadata=str(job/'missing.json'))
    (job/'annotations.json').write_text(json.dumps({'rows':[row]}))
    return source,sequence,job


class DatasetLabelTests(unittest.TestCase):
    def test_complete_export_and_binary_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,seq,job=job_fixture(Path(tmp));original={p:p.read_bytes() for p in seq.rglob('*') if p.is_file()}
            report=export_labels(job);root=seq/'labels';self.assertEqual(report['protected_frames'],0)
            self.assertEqual(set(p.name for p in root.iterdir()),{'classes.json','image_2','lidar'})
            np.testing.assert_array_equal(np.fromfile(root/'lidar/000000.label',dtype='<u4'),[7,255])
            image=json.loads((root/'image_2/000000.json').read_text());self.assertEqual(image['instances'][0]['track_id'],1);self.assertEqual(image['detections'][0]['class_id'],1)
            self.assertEqual(image['detections'][0]['detector_class_id'],0)
            with np.load(root/'image_2/000000.npz') as z:
                np.testing.assert_array_equal(np.unpackbits(z['masks_packed'],axis=1,count=4,bitorder='little'),[[0,1,0,0]])
                self.assertIn('semantic_confidence',z)
            lidar=json.loads((root/'lidar/000000.json').read_text());self.assertEqual(lidar['persons'][0]['supervision'],'xy');self.assertFalse(lidar['persons'][0]['box_3d_valid']);self.assertEqual(lidar['vehicles'][0]['box_3d'],[5,6,1,4,2,1.5,.2]);self.assertFalse(lidar['manual_ground_truth'])
            self.assertEqual(original,{p:p.read_bytes() for p in original})
            again=export_labels(job);self.assertEqual(again['frames'][0]['image'],'reused');self.assertEqual(again['frames'][0]['lidar'],'reused')

    def test_reviewed_or_manually_modified_bundle_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,seq,job=job_fixture(Path(tmp));export_labels(job);root=seq/'labels'
            image=root/'image_2/000000.json';d=json.loads(image.read_text());d['review_status']='reviewed';image.write_text(json.dumps(d));before=image.read_bytes()
            cloud=root/'lidar/000000.label';cloud.write_bytes(np.array([3,4],dtype='<u4').tobytes())
            report=export_labels(job);self.assertEqual(report['protected_frames'],1);self.assertEqual(image.read_bytes(),before);np.testing.assert_array_equal(np.fromfile(cloud,dtype='<u4'),[3,4])

    def test_changed_automatic_labels_keep_previous_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,seq,job=job_fixture(Path(tmp));export_labels(job);path=seq/'labels/image_2/000000.png';before=path.read_bytes()
            Image.fromarray(np.ones((2,2),np.uint8)).save(job/'masks/000000.png')
            report=export_labels(job);self.assertNotEqual(path.read_bytes(),before);self.assertTrue(Path(report['versions']).is_dir());self.assertEqual((Path(report['versions'])/'image_2/000000/000000.png').read_bytes(),before)

    def test_lmdb_labels_go_to_converted_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,seq,job=job_fixture(Path(tmp),kitti=False);report=export_labels(job)
            self.assertEqual(Path(report['output']),seq/'labels');self.assertFalse((source/'labels').exists())

    def test_lmdb_publishes_labels_to_explicit_public_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,seq,job=job_fixture(root,kitti=False);public=root/'public_dataset';seq.rename(public)
            owner=json.loads((job/'job.json').read_text());owner['dataset_output']=str(public);(job/'job.json').write_text(json.dumps(owner))
            result=export_labels(job)
            self.assertEqual(Path(result['output']),public/'labels')
            self.assertTrue((public/'labels/image_2/000000.png').is_file())
            self.assertTrue((public/'labels/lidar/000000.label').is_file())
            self.assertFalse((job/'kitti/labels').exists())

    def test_original_frame_id_maps_back_to_dataset_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,seq,job=job_fixture(Path(tmp))
            with np.load(seq/'metadata.npz') as z:meta={k:z[k] for k in z.files}
            meta['global_frame_ids']=np.array([63]);np.savez_compressed(seq/'metadata.npz',**meta)
            d=json.loads((job/'annotations.json').read_text());d['rows'][0]['frame_id']=63;(job/'annotations.json').write_text(json.dumps(d))
            export_labels(job);self.assertTrue((seq/'labels/image_2/000000.png').is_file());self.assertFalse((seq/'labels/image_2/000063.png').exists())
            self.assertEqual(json.loads((seq/'labels/lidar/000000.json').read_text())['source_frame_id'],63)

    def test_bad_point_order_rejected_and_other_camera_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,seq,job=job_fixture(Path(tmp));export_labels(job)
            camera=seq/'image_3';camera.mkdir();(camera/'000000.png').write_bytes((seq/'image_2/000000.png').read_bytes())
            owner=json.loads((job/'job.json').read_text());owner['camera']='image_3';(job/'job.json').write_text(json.dumps(owner))
            report=export_labels(job);self.assertTrue((seq/'labels/image_3/000000.json').exists());self.assertEqual(report['frames'][0]['lidar'],'protected_other_camera')
            path=job/'labels/000000.npz'
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['xyz']=arrays['xyz'][::-1];np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'点顺序'):export_labels(job)

    def test_raw_only_catalog_does_not_create_false_empty_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root);job=root/'job'
            with patch('tools.lidar_annotator.fresh_pipeline.execute',side_effect=lambda *a,**k:stage(job,'render',1)):run(source,job,stages=['render'])
            catalog(job);self.assertFalse((job/'kitti/labels').exists());self.assertTrue((job/'label_export.json').is_file())

if __name__=='__main__':unittest.main()
