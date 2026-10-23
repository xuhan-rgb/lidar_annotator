import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .stage_cache import complete, reuse
from .frame_selection import parse_frames, debug_output
from .fresh_pipeline import stage
from .cli import REPO


class IncrementalWorkflowTests(unittest.TestCase):
    def test_frame_selection_supports_ranges_and_original_ids(self):
        self.assertEqual(parse_frames('60-63, 79 824 61'),[60,61,62,63,79,824])
        self.assertIsNone(parse_frames(''))
        for invalid in ('9-3','-1','0:10','a'):
            with self.assertRaises(ValueError):parse_frames(invalid)

    def test_temporary_debug_path_is_stable_and_source_specific(self):
        from .frame_selection import temporary_debug_output
        a=temporary_debug_output('/tmp/source-a',[63,79])
        self.assertEqual(a,temporary_debug_output('/tmp/source-a',[79,63]))
        self.assertNotEqual(a,temporary_debug_output('/tmp/source-b',[63,79]))
        self.assertNotEqual(a,temporary_debug_output('/tmp/source-a',[63]))
        from .storage import directory
        self.assertTrue(a.is_relative_to(directory('jobs')/'debug'))

    def test_debug_subset_isolated_from_existing_full_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'job.json').write_text(json.dumps(dict(frame_ids=list(range(100)))))
            subset=debug_output(root,parse_frames('60-63'))
            self.assertEqual(subset.parent,root/'debug');self.assertEqual(debug_output(root,parse_frames('60-63')),subset);self.assertEqual(debug_output(root,None),root)
            self.assertEqual(json.loads((root/'job.json').read_text())['frame_ids'],list(range(100)))

    def test_downstream_receipt_reuses_only_unchanged_complete_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);label=job/'labels/000063.npz';label.parent.mkdir();label.write_bytes(b'labels');(job/'annotations.json').write_text('{}')
            complete(job,'catalog');self.assertTrue(reuse(job,'catalog'))
            with patch('tools.lidar_annotator.fresh_pipeline._stage') as execute,patch('tools.lidar_annotator.dataset_labels.export_labels') as export:
                stage(job,'catalog',1);execute.assert_not_called();export.assert_called_once_with(job)
            label.write_bytes(b'corrupted');self.assertFalse(reuse(job,'catalog'))
            self.assertFalse(reuse(job,'catalog',force=True))

    def test_actor_masks_not_image_segmented_again(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);(job/'recipe.json').write_text((REPO/'tools/autolabel/zgc_mask_recipe_v1/recipe.json').read_text())
            from .fresh_pipeline import _stage
            with patch('tools.lidar_annotator.actor_video.run') as video,patch('tools.lidar_annotator.fresh_semantics.run') as image:
                _stage(job,'semantic',1)
            video.assert_called_once();image.assert_called_once()

    def test_person_support_changes_invalidate_render_and_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);(job/'review').mkdir();(job/'review/index.html').write_text('review')
            (job/'labels').mkdir();(job/'labels/000000.npz').write_bytes(b'label')
            metadata=job/'temporal/map_review/sam3_full_video/interior_half_review/consistent_tracks/metadata/000000_I7.json'
            metadata.parent.mkdir(parents=True);metadata.write_text('{"selected_support_ids":[0]}')
            for name in ('render','catalog'):
                complete(job,name);self.assertTrue(reuse(job,name))
            metadata.write_text('{"selected_support_ids":[0,1]}')
            for name in ('render','catalog'):self.assertFalse(reuse(job,name))

    def test_map_refresh_does_not_invalidate_render_but_visualization_change_does(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            review = job / 'review'
            visualization = review / 'visualization'
            optimized_map = review / 'optimized_map'
            visualization.mkdir(parents=True)
            optimized_map.mkdir()
            (review / 'index.html').write_text('index')
            (review / 'report.json').write_text('{}')
            image_source = job / 'source.png'
            image_source.write_bytes(b'image')
            image = visualization / '000001.png'
            image.symlink_to(image_source)
            map_cache = optimized_map / 'cache.json'
            map_cache.write_text('map v1')

            complete(job, 'render')
            map_cache.write_text('map v2')
            self.assertTrue(reuse(job, 'render'))

            image_source.write_bytes(b'updated image')
            self.assertFalse(reuse(job, 'render'))


if __name__ == '__main__':
    unittest.main()
