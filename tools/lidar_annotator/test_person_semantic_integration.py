"""Cross-path regression checks for person semantic support filtering."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from .display_range import DEFAULT_LIMITS
from .dataset_labels import _lidar_bundle
from .fresh_pipeline import catalog
from .live_preview import fingerprint, load_frame, semantic_point_labels
from .optimized_map import build_optimized_map
from .recipe_config import default_recipe
from .track_preview import load_targets


def fixture(root):
    """One frame with exclusive person support, an overlap, and non-person labels."""
    root = Path(root)
    raw = root / 'raw'
    source = root / 'data/labels/zgc'
    tracks = root / 'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks'
    visual = root / 'data/visual_instances'
    for directory in (raw / 'frames', source / 'frames', source / 'image_masks',
                      visual, tracks / 'metadata', root / 'temporal/map_review/history_fit/bev_dedup'):
        directory.mkdir(parents=True, exist_ok=True)

    xyz = np.array([
        [0.00, 0.00, 5.0], [0.05, 0.00, 5.0], [0.10, 0.00, 5.1],
        [0.00, 0.05, 4.9], [-0.05, 0.00, 5.0], [0.00, -0.05, 5.0],
        [0.08, 0.04, 5.0], [-0.08, 0.04, 5.0],
        [0.70, 0.00, 5.0], [2.00, 0.00, 5.0],
        [0.12, 0.00, 5.0],  # shared-mask point, excluded by mask exclusivity
        [0.03, 0.03, 5.0], # original class 3 inside the person mask
    ], dtype=np.float32)
    n = len(xyz)
    uv = np.tile([4.0, 4.0], (n, 1)).astype(np.float32)
    uv[10] = [5.0, 4.0]
    labels = np.array([1] * 11 + [3], dtype=np.uint8)
    arrays = dict(xyz=xyz, pixels_uv=uv, visible=np.ones(n, bool),
                  camera_depth=xyz[:, 2].copy(), point_raw_labels=labels,
                  cam_intrinsic=np.array([[10, 0, 4], [0, 10, 4], [0, 0, 1]], float),
                  cam_extrinsic=np.eye(4))
    np.savez_compressed(source / 'frames/000000.npz', **arrays)
    np.savez_compressed(raw / 'frames/000000.npz', **arrays)
    image_path = raw / 'image.png'
    Image.new('RGB', (8, 8), (100, 100, 100)).save(image_path)
    row = dict(path='frames/000000.npz', frame_id=0, original_frame_id=63, local_frame_id=0,
               image_path=str(image_path))
    for directory in (raw, source):
        (directory / 'index.json').write_text(json.dumps({'rows': [row]}))

    semantic = np.full((8, 8), 255, np.uint8)
    semantic[4, 4] = 1
    semantic[4, 5] = 1
    Image.fromarray(semantic).save(source / 'image_masks/000000.png')
    masks = np.zeros((2, 8, 8), bool)
    masks[0, 4, 4:6] = True
    masks[1, 4, 5] = True
    np.savez_compressed(visual / '000000.npz', mask_shape=[8, 8],
                        masks_packed=np.packbits(masks.reshape(2, -1), axis=1, bitorder='little'))
    (visual / '000000.json').write_text(json.dumps({'records': [
        {'prompt': 'person', 'track_id': 7, 'bbox_xyxy': [4, 4, 6, 5]},
        {'prompt': 'person', 'track_id': 9, 'bbox_xyxy': [5, 4, 6, 5]},
    ]}))
    (root / 'data/person/metadata').mkdir(parents=True)
    (root / 'data/person/metadata/000000.json').write_text(json.dumps({'instances': [
        dict(sam_source_instance_id=7, status='accepted_xy_from_all_exclusive_mask_points',
             person_class_confirmed=True, yolo_match={'score': .9, 'bbox_xyxy': [3, 3, 7, 6]}),
    ]}))
    video = root / 'temporal/map_review/sam3_full_video'
    (video / 'native_to_display_ids.json').write_text(json.dumps({'7': 7, '9': 9}))
    centers = tracks / 'centers_only.npz'
    np.savez_compressed(centers, frame_ids=[0], sam_video_ids=[7], position_available=[True],
                        position_source=np.array(['observed']),
                        original_point_lidar=[[0, 0, 5]], corrected_point_lidar=[[0, 0, 5]])
    # Point 8 is outside the geometric core but is explicitly saved for this
    # frame and identity; point 9 is a remote, unsaved outlier.
    (tracks / 'metadata/000000_I7.json').write_text(json.dumps({
        'frame': 0, 'sam_video_id': 7, 'position_available': True,
        'selected_support_ids': [8],
    }))
    poses = root / 'temporal/poses.npz'
    np.savez_compressed(poses, transforms=np.eye(4)[None], segments=[0])
    vehicles = root / 'temporal/map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
    np.savez_compressed(vehicles, box_3d_map=np.empty((0, 7)), track_ids=np.empty(0, int))
    return dict(output=str(root)), arrays, tracks


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PersonSemanticIntegrationTests(unittest.TestCase):
    def test_load_targets_uses_exclusive_mask_and_saved_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, arrays, tracks = fixture(tmp)
            root = Path(context['output'])
            paths = dict(instance_masks=root / 'data/visual_instances/000000.npz',
                         track_masks=tracks.parent / 'masks/000000.npz',
                         tracks=tracks / 'centers_only.npz', local=0, mask=root / 'data/labels/zgc/image_masks/000000.png')
            records = json.loads((root / 'data/visual_instances/000000.json').read_text())['records']
            records[0]['display_id'] = 7
            records[1]['display_id'] = 9
            targets = load_targets(paths, arrays, records, DEFAULT_LIMITS)
            person, overlap = targets

            self.assertIn(8, person['semantic_selection']['kept_indices'])
            self.assertIn(9, person['semantic_selection']['rejected_indices'])
            self.assertNotIn(10, person['indices'])
            self.assertEqual(len(overlap['indices']), 0)
            self.assertTrue(person['semantic_selection']['saved_support_used'])

            # Tight display bounds hide some points, but don't alter semantic decisions.
            narrow = load_targets(paths, arrays, records, (20., -3., 4.95))
            np.testing.assert_array_equal(narrow[0]['rejected_indices'], person['rejected_indices'])
            np.testing.assert_array_equal(narrow[0]['semantic_selection']['kept_indices'],
                                          person['semantic_selection']['kept_indices'])
            self.assertLess(len(narrow[0]['optimized_indices']), len(person['optimized_indices']))

    def test_frame_and_accumulated_map_apply_the_same_person_ignore_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, arrays, _ = fixture(tmp)
            data = load_frame(context, 63, mode='tracks')
            target = data['targets'][0]
            frame_labels = semantic_point_labels(arrays, None, data['targets'])
            np.testing.assert_array_equal(frame_labels[target['rejected_indices']], 255)
            self.assertEqual(frame_labels[11], 3)
            self.assertEqual(frame_labels[8], 1)  # saved support remains semantic person

            map_data = build_optimized_map(context, spacing=0)
            with np.load(map_data['archive'], allow_pickle=False) as archive:
                source_labels = archive['point_raw_labels']
                map_frame_ids = archive['source_frame_ids']
            np.testing.assert_array_equal(map_frame_ids, np.full(len(arrays['xyz']), 63))
            np.testing.assert_array_equal(source_labels, frame_labels)

    def test_small_support_or_missing_center_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, arrays, tracks = fixture(tmp)
            root = Path(context['output'])
            paths = dict(instance_masks=root / 'data/visual_instances/000000.npz',
                         track_masks=tracks.parent / 'masks/000000.npz',
                         tracks=tracks / 'centers_only.npz', local=0, mask=root / 'data/labels/zgc/image_masks/000000.png')
            records = json.loads((root / 'data/visual_instances/000000.json').read_text())['records']
            records[0]['display_id'] = 7
            records[1]['display_id'] = 9
            sparse = dict(arrays, point_raw_labels=arrays['point_raw_labels'].copy())
            sparse['point_raw_labels'][1:8] = 3
            result = load_targets(paths, sparse, records, DEFAULT_LIMITS)[0]
            self.assertTrue(result['semantic_selection']['review_required'])
            self.assertEqual(len(result['rejected_indices']), 0)

            # An explicit per-frame unavailable-position record also fails open.
            saved_path = tracks / 'metadata/000000_I7.json'
            saved = json.loads(saved_path.read_text())
            saved['position_available'] = False
            saved_path.write_text(json.dumps(saved))
            result = load_targets(paths, arrays, records, DEFAULT_LIMITS)[0]
            self.assertTrue(result['semantic_selection']['review_required'])
            self.assertEqual(len(result['rejected_indices']), 0)

    def test_catalog_and_public_lidar_bundle_keep_the_shared_person_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, arrays, tracks = fixture(tmp)
            root = Path(context['output'])
            backend = dict(source_index=str(root / 'data/labels/zgc/index.json'),
                           temporal_output=str(root / 'temporal'), pose_npz=str(root / 'temporal/poses.npz'),
                           coordinate_assumption='synthetic test coordinates',
                           person_root=str(root / 'data/person'), vehicle_root=str(root / 'data/vehicle'))
            (root / 'backend.json').write_text(json.dumps(backend))
            (root / 'recipe.json').write_text(json.dumps(default_recipe()))

            source_npz = root / 'data/labels/zgc/frames/000000.npz'
            centers_npz = tracks / 'centers_only.npz'
            pose_npz = root / 'temporal/poses.npz'
            before = {path: sha256(path) for path in (source_npz, centers_npz, pose_npz)}
            # Catalog owns the semantic selection path; skip only KITTI conversion/export.
            with patch('tools.lidar_annotator.dataset_labels.export_labels'):
                catalog(root)

            label_npz = root / 'labels/000063.npz'
            with np.load(label_npz, allow_pickle=False) as z:
                catalog_labels = z['point_raw_labels'].copy()
                audit = json.loads(str(z['person_semantic_filter_json'].item()))
                np.testing.assert_array_equal(z['person_semantic_ignored_indices'], [9])
            self.assertEqual(audit['ignored_indices'], [9])
            self.assertIn(9, audit['instances'][0]['ignored_indices'])

            points = np.c_[arrays['xyz'], np.ones(len(arrays['xyz']), np.float32)].astype('<f4')
            points_path = root / 'original_points.bin'
            points.tofile(points_path)
            bundle, artifacts = _lidar_bundle(
                dict(label_path=label_npz), 'image_2', 0, 63, points_path)
            public_label = root / 'public.label'
            public_label.write_bytes(artifacts['.label'])
            public_labels = np.fromfile(public_label, dtype='<u4')
            self.assertEqual(bundle['person_semantic_filter']['ignored_indices'], [9])

            frame_semantics_labels = semantic_point_labels(arrays, None,
                load_frame(context, 63, mode='tracks')['targets'])
            map_data = build_optimized_map(context, spacing=0)
            with np.load(map_data['archive'], allow_pickle=False) as z:
                map_labels = z['point_raw_labels'].copy()
            np.testing.assert_array_equal(catalog_labels, frame_semantics_labels)
            np.testing.assert_array_equal(map_labels, frame_semantics_labels)
            np.testing.assert_array_equal(public_labels, frame_semantics_labels.astype('<u4'))
            self.assertEqual(np.flatnonzero(frame_semantics_labels == 255).tolist(), [9])
            self.assertEqual({path: sha256(path) for path in before}, before)

    def test_support_metadata_changes_fingerprint_and_next_frame_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, _, tracks = fixture(tmp)
            before = load_frame(context, 63, mode='tracks')['targets'][0]
            self.assertIn(8, before['semantic_selection']['kept_indices'])
            fingerprint_before = fingerprint(context, 63)

            metadata = tracks / 'metadata/000000_I7.json'
            saved = json.loads(metadata.read_text())
            saved['selected_support_ids'] = []
            metadata.write_text(json.dumps(saved))

            fingerprint_after = fingerprint(context, 63)
            self.assertNotEqual(fingerprint_before, fingerprint_after)
            after = load_frame(context, 63, mode='tracks')['targets'][0]
            self.assertNotIn(8, after['semantic_selection']['kept_indices'])
            self.assertIn(8, after['semantic_selection']['rejected_indices'])


if __name__ == '__main__':
    unittest.main()
