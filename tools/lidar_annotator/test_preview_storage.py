import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import fresh_pipeline as pipeline


class PreviewStorageTests(unittest.TestCase):
    def test_link_preview_uses_relative_link_and_replaces_file_or_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'review_local/visualization/000003.png'
            destination = root / 'review/visualization/000042.png'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'original preview bytes')
            destination.parent.mkdir(parents=True)

            destination.write_bytes(b'old copied preview')
            pipeline._link_preview(source, destination)
            self.assertTrue(destination.is_symlink())
            self.assertFalse(Path(destination.readlink()).is_absolute())
            self.assertEqual(source.read_bytes(), destination.read_bytes())

            other = root / 'review_local/visualization/000004.png'
            other.write_bytes(b'new preview bytes')
            destination.unlink()
            destination.symlink_to(Path('some-old-target.png'))
            pipeline._link_preview(other, destination)
            self.assertTrue(destination.is_symlink())
            self.assertFalse(Path(destination.readlink()).is_absolute())
            self.assertEqual(other.read_bytes(), destination.read_bytes())

    def test_optimized_render_links_local_files_to_original_frame_names(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            source = job / 'data/labels/zgc'
            source.mkdir(parents=True)
            (source / 'index.json').write_text(json.dumps({'rows': [
                {'frame_id': 3, 'original_frame_id': 42, 'path': 'frames/000003.npz'}]}))
            local = job / 'review_local/visualization'
            optimized = job / 'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
            optimized.parent.mkdir(parents=True)
            np.savez_compressed(optimized, centers=np.zeros((1, 3), np.float32))
            (job / 'backend.json').write_text(json.dumps({
                'source_index': str(source / 'index.json'), 'temporal_output': str(job / 'temporal')}))

            contents = {'000003.png': b'png bytes', '000003.json': b'{"review":true}'}
            def fake_backend(_job, _script, *_args):
                local.mkdir(parents=True)
                for name, data in contents.items():
                    (local / name).write_bytes(data)

            with patch.object(pipeline, 'backend', side_effect=fake_backend):
                pipeline.render(job, 1)

            for suffix, expected in (('.png', contents['000003.png']),
                                     ('.json', contents['000003.json'])):
                public_path = job / 'review/visualization' / ('000042' + suffix)
                canonical_path = local / ('000003' + suffix)
                self.assertTrue(public_path.is_symlink())
                self.assertEqual(expected, public_path.read_bytes())
                self.assertEqual(expected, canonical_path.read_bytes())
                self.assertFalse(Path(public_path.readlink()).is_absolute())


if __name__ == '__main__':
    unittest.main()
