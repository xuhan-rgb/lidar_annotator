import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .class_semantic_cache import load_class_cache, save_class_cache


class ClassSemanticCacheTests(unittest.TestCase):
    def test_randomized_planes_round_trip_exactly(self):
        rng = np.random.default_rng(412)
        floats = rng.standard_normal((31, 27), dtype=np.float32)
        floats[0, 0] = np.float32(-0.0)
        planes = {'building': floats, 'road': rng.random((31, 27)) > .7}
        metadata = {'frame': '000123', 'version': 1}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'frame.npz'
            save_class_cache(path, planes, metadata)
            loaded, loaded_metadata = load_class_cache(path, floats.shape)
        self.assertEqual(metadata, loaded_metadata)
        for name, source in planes.items():
            self.assertEqual(source.dtype, loaded[name].dtype)
            self.assertEqual(source.shape, loaded[name].shape)
            self.assertEqual(source.tobytes(), loaded[name].tobytes())

    def test_all_zero_planes_round_trip(self):
        planes = {'empty_float': np.zeros((8, 9), np.float32),
                  'empty_bool': np.zeros((8, 9), bool)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'empty.npz'
            save_class_cache(path, planes, {})
            loaded, metadata = load_class_cache(path, (8, 9))
        self.assertEqual({}, metadata)
        for name in planes:
            self.assertEqual(planes[name].tobytes(), loaded[name].tobytes())

    def test_corruption_and_shape_mismatch_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.npz'
            save_class_cache(path, {'road': np.ones((3, 4), np.float32)}, {'x': 1})
            self.assertIsNone(load_class_cache(path, (4, 3)))
            path.write_bytes(path.read_bytes() + b'corruption')
            self.assertIsNone(load_class_cache(path, (3, 4)))

    def test_modified_receipt_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.npz'
            save_class_cache(path, {'road': np.ones((3, 4), np.float32)}, {'x': 1})
            receipt_path = path.with_suffix('.npz.json')
            receipt = json.loads(receipt_path.read_text())
            receipt['metadata']['x'] = 2
            receipt_path.write_text(json.dumps(receipt))
            self.assertIsNone(load_class_cache(path, (3, 4)))

    def test_truncated_archive_and_missing_receipt_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.npz'
            save_class_cache(path, {'road': np.ones((3, 4), np.float32)}, {})
            path.write_bytes(path.read_bytes()[:12])
            self.assertIsNone(load_class_cache(path, (3, 4)))
            path.unlink()
            receipt_path = path.with_suffix('.npz.json')
            receipt_path.unlink()
            self.assertIsNone(load_class_cache(path, (3, 4)))


if __name__ == '__main__':
    unittest.main()
