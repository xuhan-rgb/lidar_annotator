"""Explicit person-class decisions gate semantic support before geometry review."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from .optimized_map import load_optimized_map, build_optimized_map

from .display_range import DEFAULT_LIMITS
from .test_person_semantic_integration import fixture
from .track_preview import load_targets, person_support_audit


class PersonClassFilterTests(unittest.TestCase):
    def load(self, root):
        context, arrays, tracks = fixture(root)
        root = Path(context['output'])
        metadata = root / 'data/person/metadata/000000.json'
        paths = dict(instance_masks=root / 'data/visual_instances/000000.npz',
                     track_masks=tracks.parent / 'masks/000000.npz',
                     tracks=tracks / 'centers_only.npz', local=0,
                     mask=root / 'data/labels/zgc/image_masks/000000.png',
                     person_metadata=metadata,
                     id_map=root / 'temporal/map_review/sam3_full_video/native_to_display_ids.json')
        records = json.loads((root / 'data/visual_instances/000000.json').read_text())['records']
        records[0]['display_id'] = 7
        records[1]['display_id'] = 9
        return paths, arrays, records, metadata, tracks

    def test_explicitly_unconfirmed_person_rejects_all_its_person_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arrays, records, metadata, tracks = self.load(tmp)
            value = json.loads(metadata.read_text())
            value['instances'][0]['person_class_confirmed'] = False
            value['instances'][0]['yolo_match'] = None
            metadata.write_text(json.dumps(value))
            # Missing center currently causes the geometry filter to fail open.
            with np.load(tracks / 'centers_only.npz') as archive:
                center_data = {key: archive[key] for key in archive.files}
            center_data['position_available'][0] = False
            center_data['corrected_point_lidar'] = center_data['corrected_point_lidar'].astype(float)
            center_data['corrected_point_lidar'][0] = np.nan
            np.savez_compressed(tracks / 'centers_only.npz', **center_data)
            targets = load_targets(paths, arrays, records, DEFAULT_LIMITS)
            target = targets[0]
            self.assertIsNone(targets[1]['person_class_confirmed'])  # metadata for ID 7 cannot leak to ID 9
            np.testing.assert_array_equal(target['rejected_indices'], np.arange(10))
            self.assertEqual(target['semantic_selection']['status'], 'unconfirmed_person_instance')
            self.assertTrue(target['semantic_selection']['review_required'])
            self.assertIs(target['person_class_confirmed'],False)
            np.testing.assert_array_equal(target['semantic_selection']['kept_indices'], [])
            self.assertEqual(person_support_audit([target])['instances'][0]['class_confirmed'], False)

    def test_confirmed_sparse_and_legacy_missing_flag_remain_conservative(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arrays, records, metadata, _ = self.load(tmp)
            value = json.loads(metadata.read_text())
            value['instances'][0]['person_class_confirmed'] = True
            value['instances'][0]['yolo_match']['score'] = .9
            metadata.write_text(json.dumps(value))
            sparse = dict(arrays, point_raw_labels=arrays['point_raw_labels'].copy())
            sparse['point_raw_labels'][1:8] = 3
            target = load_targets(paths, sparse, records, DEFAULT_LIMITS)[0]
            self.assertEqual(target['person_class_confirmed'], True)
            self.assertTrue(target['semantic_selection']['review_required'])
            self.assertEqual(len(target['rejected_indices']), 0)

            value['instances'][0]['yolo_match']['score'] = .2
            metadata.write_text(json.dumps(value))
            target = load_targets(paths, arrays, records, DEFAULT_LIMITS)[0]
            self.assertIs(target['person_class_confirmed'],False)
            self.assertEqual(target['semantic_selection']['status'], 'unconfirmed_person_instance')

            value['instances'][0].pop('person_class_confirmed')
            metadata.write_text(json.dumps(value))
            target = load_targets(paths, arrays, records, DEFAULT_LIMITS)[0]
            self.assertIsNone(target['person_class_confirmed'])
            self.assertNotEqual(target['semantic_selection']['status'], 'unconfirmed_person_instance')

    def test_identity_mapping_is_frame_local_and_static_class_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arrays, records, metadata, _ = self.load(tmp)
            value = json.loads(metadata.read_text())
            value['instances'][0]['person_class_confirmed'] = False
            value['instances'][0]['yolo_match'] = None
            metadata.write_text(json.dumps(value))
            target = load_targets(paths, arrays, records, DEFAULT_LIMITS)[0]
            self.assertNotIn(11, target['rejected_indices'])
            self.assertEqual(arrays['point_raw_labels'][11], 3)

    def test_display_only_record_uses_flat_native_id_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arrays, records, metadata, _ = self.load(tmp)
            native_id=4082546627380772871
            metadata.write_text(json.dumps({'instances':[dict(sam_source_instance_id=native_id,person_class_confirmed=False,yolo_match=None)]}))
            paths['id_map'].write_text(json.dumps({str(native_id):7}))
            records[0].pop('track_id')
            target=load_targets(paths,arrays,records,DEFAULT_LIMITS)[0]
            self.assertIs(target['person_class_confirmed'],False)
            np.testing.assert_array_equal(target['rejected_indices'],np.arange(10))

    def test_class_confirmation_change_invalidates_saved_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, arrays, records, metadata, _ = self.load(tmp)
            context={'output':tmp}
            first=load_optimized_map(context,spacing=0)
            value=json.loads(metadata.read_text())
            value['instances'][0]['person_class_confirmed']=False
            value['instances'][0]['yolo_match']=None
            metadata.write_text(json.dumps(value))
            updated=load_optimized_map(context,spacing=0)
            self.assertFalse(updated['cache_reused'])
            self.assertNotEqual(first['labels'].tolist(),updated['labels'].tolist())
            np.testing.assert_array_equal(updated['labels'][:10],np.full(10,255))
            self.assertEqual(updated['labels'][11],3)

    def test_map_reads_each_frames_own_class_confirmation(self):
        from .test_optimized_map import fixture as map_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,_,_=map_fixture(root)
            metadata=root/'data/person/metadata';metadata.mkdir(parents=True)
            for frame,confirmed in enumerate((True,False)):
                (metadata/f'{frame:06d}.json').write_text(json.dumps({'instances':[dict(
                    sam_source_instance_id=7,person_class_confirmed=confirmed,
                    yolo_match={'score':.9} if confirmed else None)]}))
            data=build_optimized_map(context,spacing=0)
            np.testing.assert_array_equal(data['labels'],[1,1,3,255,255,255,3,255])
