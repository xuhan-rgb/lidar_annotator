import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalize_person_positions import load_positions, run


class FinalizePersonPositionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.temporal = self.root / "temporal"
        self.tracks = self.temporal / "map_review/sam3_full_video/interior_half_review/consistent_tracks"
        self.tracks.mkdir(parents=True)
        self.video = self.temporal / "map_review/sam3_full_video"
        self.person = self.root / "person"
        (self.person / "metadata").mkdir(parents=True)
        source = self.root / "source"
        (source / "frames").mkdir(parents=True)
        self.source_index = source / "index.json"
        self.source_index.write_text(json.dumps({"rows": [{"path": "frames/000099.npz", "local_frame_id": 0}]}))
        self.pose = self.root / "poses.npz"
        angle = np.pi / 2
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        pose[:3, 3] = [4, 5, 2]
        np.savez(self.pose, transforms=pose[None])
        np.savez(source / "frames/000099.npz", xyz=np.array([[1., 0., 0.]] * 6))
        (self.person / "metadata/000000.json").write_text(json.dumps({"instances": [{
            "sam_source_instance_id": "native", "exclusive_support_points": list(range(6)),
            "yolo_match": {"score": .5}}]}))
        (self.video / "native_to_display_ids.json").write_text(json.dumps({"native": 7}))
        self.original_local = np.array([[0., 0., 3.]])
        self.reference = np.array([[4., 6.]])
        self.diagnostic = self.tracks / "centers_only.npz"
        np.savez(self.diagnostic, frame_ids=[0], instance_ids=[1], sam_video_ids=[7], timestamp_ns=[12],
                 original_point_lidar=[[1., 2., 3.]], corrected_point_lidar=self.original_local,
                 corrected_xy_map=[[0., 0.]], reference_xy_map=self.reference,
                 position_available=[True], position_source=["trajectory_regularized_camera_depth"],
                 camera_depth=[3.], selected_support_counts=[0])
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"source_index": str(self.source_index), "pose_npz": str(self.pose),
                                           "temporal_output": str(self.temporal), "person_root": str(self.person)}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_global_xy_inverse_pose_z_and_support_policy(self):
        source_row = json.loads(self.source_index.read_text())["rows"][0]
        self.assertEqual(source_row["local_frame_id"], 0)
        self.assertEqual(Path(source_row["path"]).stem, "000099")
        before = self.diagnostic.read_bytes()
        report = run(self.config)
        result = load_positions(self.diagnostic)
        self.assertTrue(np.array_equal(self.diagnostic.read_bytes(), before))
        self.assertEqual(result["position_policy"], "global_reference_xy_v1")
        pose = np.load(self.pose)["transforms"][0]
        world = pose[:3, :3] @ result["corrected_point_lidar"][0] + pose[:3, 3]
        np.testing.assert_allclose(world, [4., 6., 5.])
        self.assertEqual(result["selected_support_counts"].tolist(), [6])
        self.assertEqual(result["track_only"].tolist(), [False])
        self.assertEqual(result["ray_position_source"].tolist(), ["trajectory_regularized_camera_depth"])
        self.assertEqual(result["ray_camera_depth"].tolist(), [3.])
        self.assertNotIn("camera_depth", result)
        self.assertEqual(report["records"], 1)

    def test_sparse_current_support_is_track_only(self):
        (self.person / "metadata/000000.json").write_text(json.dumps({"instances": [{
            "sam_source_instance_id": "native", "exclusive_support_points": list(range(4))}]}))
        run(self.config)
        result = load_positions(self.diagnostic)
        self.assertEqual(result["selected_support_counts"].tolist(), [4])
        self.assertEqual(result["track_only"].tolist(), [True])

    def test_same_content_rewrites_reuse_and_changed_pose_invalidates(self):
        run(self.config)
        final = self.tracks / "final_centers.npz"
        mtime = final.stat().st_mtime_ns
        index_contents = self.source_index.read_text()
        self.source_index.write_text(index_contents)
        self.pose.touch()
        load_positions(self.diagnostic)
        run(self.config)
        self.assertEqual(final.stat().st_mtime_ns, mtime)
        with np.load(final, allow_pickle=False) as z:
            old_inputs = str(z["inputs_json"].item())
        with np.load(self.pose) as z:
            transforms = z["transforms"].copy()
        transforms[0, 0, 3] += 1
        np.savez(self.pose, transforms=transforms)
        with self.assertRaisesRegex(ValueError, "最终行人位置已过期，请点击开始执行更新下游结果"):
            load_positions(self.diagnostic)
        run(self.config)
        with np.load(final, allow_pickle=False) as z:
            self.assertNotEqual(str(z["inputs_json"].item()), old_inputs)
        load_positions(self.diagnostic)

    def test_unavailable_nan_position_has_finite_report_and_ray_diagnostics(self):
        np.savez(self.diagnostic, frame_ids=[0], sam_video_ids=[7], original_point_lidar=[[1., 2., 3.]],
                 corrected_point_lidar=[[0., 0., 3.]], reference_xy_map=[[np.nan, np.nan]],
                 position_available=[False], position_source=["ignored_camera_ray_continuity_conflict"],
                 camera_depth=[np.nan])
        report = run(self.config)
        self.assertEqual(report["max_xy_displacement_m"], 0.)
        json.dumps(report, allow_nan=False)
        result = load_positions(self.diagnostic)
        self.assertEqual(result["corrected_point_lidar"].shape, (1, 3))
        self.assertEqual(result["position_available"].tolist(), [False])
        self.assertEqual(result["ray_position_source"].tolist(), ["ignored_camera_ray_continuity_conflict"])

    def test_empty_population_keeps_shaped_arrays_and_finite_report(self):
        np.savez(self.diagnostic, frame_ids=np.array([], dtype=int), sam_video_ids=np.array([], dtype=int),
                 original_point_lidar=np.empty((0, 3)), corrected_point_lidar=np.empty((0, 3)),
                 reference_xy_map=np.empty((0, 2)), position_available=np.array([], dtype=bool),
                 position_source=np.array([], dtype="U10"), camera_depth=np.array([], dtype=float))
        report = run(self.config)
        result = load_positions(self.diagnostic)
        self.assertEqual(result["corrected_point_lidar"].shape, (0, 3))
        self.assertEqual(result["reference_xy_map"].shape, (0, 2))
        self.assertEqual(report["records"], 0)
        self.assertEqual(report["max_xy_displacement_m"], 0.)
        json.dumps(report, allow_nan=False)

    def test_legacy_loader_marks_sparse_support_and_source(self):
        self.diagnostic.unlink()
        np.savez(self.diagnostic, frame_ids=[0, 1], selected_support_counts=[4, 5],
                 position_source=["x", "x"])
        result = load_positions(self.diagnostic)
        self.assertEqual(result["position_policy"], "legacy_camera_ray")
        self.assertEqual(result["track_only"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
