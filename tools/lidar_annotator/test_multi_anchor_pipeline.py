import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .cli import commands
from .fresh_pipeline import _stage


class MultiAnchorPipelineTests(unittest.TestCase):
    def test_cli_optimize_runs_geometry_refinement_between_dedup_and_tracks(self):
        config = dict(python='/runtime/python', backend_config='/job/backend.json')
        names = [Path(command[1]).name for command in commands(config, 'optimize')]
        self.assertEqual(names, [
            'visualize_vehicle_map.py', 'fit_vehicle_map.py',
            'deduplicate_vehicle_map.py', 'refine_vehicle_map.py',
            'run_person_consistent_tracks.py',
            'finalize_person_positions.py',
        ])

    def test_fresh_optimize_calls_geometry_refinement_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            with patch('tools.autolabel.task_progress.report_progress'), \
                    patch('tools.lidar_annotator.fresh_pipeline.backend', side_effect=lambda job, name, *args: calls.append(name)):
                _stage(Path(directory), 'optimize', 1)
        self.assertEqual(calls, [
            'visualize_vehicle_map.py', 'fit_vehicle_map.py',
            'deduplicate_vehicle_map.py', 'refine_vehicle_map.py',
            'run_person_consistent_tracks.py',
            'finalize_person_positions.py',
        ])
        self.assertEqual(calls.count('refine_vehicle_map.py'), 1)


if __name__ == '__main__':
    unittest.main()
