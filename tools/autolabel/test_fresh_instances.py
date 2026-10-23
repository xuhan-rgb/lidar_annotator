import json
import tempfile
import unittest

import numpy as np
from pathlib import Path

from tools.autolabel.fresh_instances import _prepare, _source_aliases, _link_processing_input, _json


class SourceIndexTest(unittest.TestCase):
    def test_infinite_obstacle_distance_is_exported_as_one_million(self):
        rows = [
            {"frame_id": 0, "source_index_record": {"min_gt_obstacle_distance_m": float("inf"), "gt_valid": False}},
            {"frame_id": 1, "source_index_record": {"min_gt_obstacle_distance_m": 2.25}},
            {"frame_id": 2},
        ]
        with tempfile.TemporaryDirectory() as folder:
            source, *_ = _prepare(Path(folder), rows)
            saved = json.loads((source / "index.json").read_text(),
                               parse_constant=lambda value: self.fail(value))
        self.assertEqual(saved["rows"][0]["source_index_record"],
                         {"min_gt_obstacle_distance_m": 1e6, "gt_valid": False})
        self.assertEqual(saved["rows"][1:], rows[1:])
        self.assertEqual(rows[0]["source_index_record"]["min_gt_obstacle_distance_m"], float("inf"))


class SourceAliasStorageTest(unittest.TestCase):
    def test_prediction_alias_follows_atomic_refresh_without_changing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'raw/prediction.json'; destination = root / 'frames/prediction.json'
            first = dict(items=[dict(score=.8, track_id=4)])
            _json(source, first); _link_processing_input(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            _json(source, dict(items=[]))
            self.assertEqual(json.loads(destination.read_text()), dict(items=[]))
            _json(destination, first)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(json.loads(source.read_text()), dict(items=[]))

    def test_complete_frame_is_shared_and_arrays_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'semantic/frame.npz'; source.parent.mkdir()
            arrays = dict(xyz=np.ones((3, 3), np.float32), cam_intrinsic=np.eye(3),
                          cam_extrinsic=np.eye(4), lidar_timestamp_ns=np.array(123),
                          point_labels=np.array([0, 1, 255], np.uint8))
            np.savez_compressed(source, **arrays); before = source.read_bytes()
            destination = root / 'consumer/frame.npz'
            _source_aliases(source, destination)
            self.assertTrue(destination.is_symlink())
            self.assertFalse(Path(destination.readlink()).is_absolute())
            self.assertEqual(destination.read_bytes(), before)
            _source_aliases(source, destination)
            self.assertEqual(source.read_bytes(), before)
            with np.load(destination) as z:
                for key, expected in arrays.items(): np.testing.assert_array_equal(z[key], expected)

    def test_missing_aliases_are_added_without_mutating_existing_link_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); previous = root / 'old.npz'; source = root / 'new.npz'
            np.savez_compressed(previous, xyz=np.ones((2, 3)), point_labels=np.ones(2, np.uint8))
            before = previous.read_bytes()
            np.savez_compressed(source, xyz=np.zeros((2, 3)), K=np.eye(3), E=np.eye(4), timestamp=np.array(456))
            destination = root / 'alias.npz'; destination.symlink_to(previous.name)
            _source_aliases(source, destination)
            self.assertEqual(previous.read_bytes(), before)
            self.assertFalse(destination.is_symlink())
            with np.load(destination) as z:
                np.testing.assert_array_equal(z['point_labels'], [0, 0])
                np.testing.assert_array_equal(z['cam_intrinsic'], np.eye(3))
                self.assertEqual(z['lidar_timestamp_ns'], 456)


if __name__ == "__main__":
    unittest.main()
