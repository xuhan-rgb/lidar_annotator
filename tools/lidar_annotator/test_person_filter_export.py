"""Person semantic filtering is carried through catalog and public export."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .fresh_pipeline import catalog
from .dataset_labels import export_labels
from .test_dataset_labels import job_fixture


class PersonFilterExportTests(unittest.TestCase):
    def test_catalog_filter_reaches_public_label_and_audit_without_moving_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sequence, job = job_fixture(Path(tmp))
            # Supply the shared selector's canonical decisions at the catalog boundary.
            targets = [dict(id=4, prompt='person', rejected_indices=np.array([0, 1]), semantic_selection=dict(
                status='person_support_filtered', review_required=False, center_xy=[1., 2.],
                saved_support_used=True, kept_indices=np.array([], dtype=int),
                rejected_indices=np.array([0, 1])))]
            arrays = np.load(job/'raw/frames/000000.npz', allow_pickle=False)
            source_arrays = {k: arrays[k] for k in arrays.files}
            arrays.close()
            source_arrays.update(point_raw_labels=np.array([2, 1], np.uint8),
                                 point_raw_confidence=np.array([.8, .7], np.float32),
                                 point_labels=np.array([2, 1], np.uint8),
                                 point_confidence=np.array([.8, .7], np.float32),
                                 supervision_policy=np.asarray('static_drivable_occupancy_v1'),
                                 person_xyz=np.array([[1., 2., 3.]]), person_track_ids=np.array([4]),
                                 person_position_source=np.array(['observed']),
                                 vehicle_box_3d=np.array([[5., 6., 1., 4., 2., 1.5, .2]]),
                                 vehicle_track_ids=np.array([3]))
            from .fresh_semantics import weak_targets, teacher_bev_from_occupancy_bits
            weak = weak_targets(source_arrays, schema='zgc_image_teacher_seven_v2')
            source_arrays.update(weak)
            bits = weak['weak_occupancy_bits']
            occupancy = np.full((320, 320), 255, np.uint8)
            for label, occ in enumerate((0, 1, 2, 5, 4, 6, 3)):
                occupancy[bits == (1 << occ)] = occ
            source_arrays['occupancy_labels'] = occupancy
            source_arrays['bev_labels'] = teacher_bev_from_occupancy_bits(bits)
            cell = np.floor((source_arrays['xyz'][1, :2] + 40) / .25).astype(int)
            cell = (int(cell[1]), int(cell[0]))
            np.savez_compressed(job/'labels/000000.npz', **source_arrays)
            np.savez_compressed(job/'raw/frames/000000.npz', **source_arrays)
            row = json.loads((job/'annotations.json').read_text())['rows'][0]
            row.update(local_frame=0, label='labels/000000.npz')
            (job/'annotations.json').write_text(json.dumps({'rows':[row]}))
            # Exercise catalog's shared-selection wiring while isolating selector behavior.
            with patch('tools.lidar_annotator.track_preview.load_targets', return_value=targets), \
                 patch('tools.lidar_annotator.fresh_pipeline.config', return_value=dict(
                     source_index=str(job/'raw/index.json'), temporal_output=str(job/'temporal'),
                     pose_npz=str(job/'temporal/poses.npz'), coordinate_assumption='test',
                     person_root=str(job/'data/person'), vehicle_root=str(job/'data/vehicle'))):
                video = job/'temporal/map_review/sam3_full_video'
                person = video/'interior_half_review/consistent_tracks'
                person.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(person/'centers_only.npz', frame_ids=np.array([0]),
                                    position_available=np.array([True]),
                                    corrected_point_lidar=np.array([[1., 2., 3.]]),
                                    sam_video_ids=np.array([4]),
                                    position_source=np.array(['observed']))
                (job/'temporal/map_review/history_fit/bev_dedup').mkdir(parents=True)
                np.savez_compressed(job/'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz',
                                    box_3d_map=source_arrays['vehicle_box_3d'], track_ids=np.array([3]))
                np.savez_compressed(job/'temporal/poses.npz', transforms=np.eye(4)[None])
                catalog(job)
            with np.load(job/'labels/000000.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['point_raw_labels'], [2, 255])
                np.testing.assert_allclose(z['point_raw_confidence'], [.8, 0])
                np.testing.assert_array_equal(z['point_labels'], [2, 255])
                np.testing.assert_allclose(z['point_confidence'], [.8, 0])
                np.testing.assert_array_equal(z['xyz'], source_arrays['xyz'])
                np.testing.assert_array_equal(z['person_xyz'], source_arrays['person_xyz'])
                np.testing.assert_array_equal(z['vehicle_box_3d'], source_arrays['vehicle_box_3d'])
                self.assertEqual(z['occupancy_labels'][cell], 2)
                self.assertEqual(z['bev_labels'][tuple(cell[::-1])], 2)
                self.assertAlmostEqual(float(z['weak_confidence'][cell]), .8, places=6)
                self.assertEqual(z['weak_occupancy_bits'][cell], 4)
            export_labels(job)
            labels = np.fromfile(sequence/'labels/lidar/000000.label', dtype='<u4')
            np.testing.assert_array_equal(labels, [2, 255])
            audit = json.loads((sequence/'labels/lidar/000000.json').read_text())['person_semantic_filter']
            self.assertEqual(audit['ignored_indices'], [1])
            self.assertEqual(audit['instances'][0]['ignored_indices'], [1])
            actually_filtered = np.flatnonzero((source_arrays['point_raw_labels'] == 1) & (labels == 255)).tolist()
            self.assertEqual(audit['ignored_indices'], actually_filtered)
            self.assertEqual(audit['instances'][0]['status'], 'person_support_filtered')


if __name__ == '__main__':
    unittest.main()
