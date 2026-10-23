"""Create stable global-reference pedestrian positions without changing diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

POLICY = "global_reference_xy_v1"


def _stamp(path):
    p = Path(path).resolve()
    try:
        return [str(p), hashlib.sha256(p.read_bytes()).hexdigest()]
    except OSError:
        return [str(p), None]


def _paths(config, diagnostic, id_map):
    return [diagnostic, config["pose_npz"], config["source_index"],
            Path(config["person_root"]) / "index.json", id_map, Path(__file__)]


def _current_inputs(paths):
    return json.dumps([_stamp(p) for p in paths], separators=(",", ":"))


def load_positions(tracks_path):
    path = Path(tracks_path)
    final = path.with_name("final_centers.npz")
    if final.is_file():
        with np.load(final, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        deps = json.loads(str(data.pop("inputs_json")))
        now = [_stamp(x[0]) for x in deps]
        if deps != now:
            raise ValueError(f"最终行人位置已过期，请点击开始执行更新下游结果（{final}）")
        data["position_policy"] = str(data["position_policy"].item())
        return data
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    data["position_policy"] = "legacy_camera_ray"
    counts = data.get("selected_support_counts")
    source = data.get("position_source")
    if counts is not None:
        data["track_only"] = counts < 5
    elif source is not None:
        data["track_only"] = source == "linear_interpolation_only"
    else:
        data["track_only"] = np.zeros(len(data.get("frame_ids", [])), dtype=bool)
    return data


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path(config["temporal_output"])
    video = root / "map_review/sam3_full_video"
    tracks = video / "interior_half_review/consistent_tracks"
    diagnostic = tracks / "centers_only.npz"
    final = tracks / "final_centers.npz"
    id_map = video / "native_to_display_ids.json"
    paths = _paths(config, diagnostic, id_map)
    inputs = _current_inputs(paths)
    if final.is_file():
        try:
            with np.load(final, allow_pickle=False) as z:
                if str(z["inputs_json"].item()) == inputs:
                    return json.loads((tracks / "final_positions_report.json").read_text())
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            pass

    with np.load(diagnostic, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    with np.load(config["pose_npz"], allow_pickle=False) as z:
        poses = z["transforms"].copy()
    source_index = Path(config["source_index"])
    rows = json.loads(source_index.read_text())["rows"]
    source_root = source_index.parent
    person_root = Path(config["person_root"])
    mapping = json.loads(id_map.read_text()) if id_map.is_file() else {}
    frames = data["frame_ids"].astype(int)
    ids = data["sam_video_ids"].astype(int)
    refs = data["reference_xy_map"].astype(float)
    old = data["corrected_point_lidar"].astype(float)
    local_final = np.full_like(old, np.nan)
    counts = np.zeros(len(frames), dtype=np.int64)
    confirmed = np.zeros(len(frames), dtype=bool)
    available = data["position_available"].astype(bool).copy()
    frame_rows = {int(row.get("local_frame_id", Path(row["path"]).stem)): row for row in rows}
    for frame in np.unique(frames):
        row = frame_rows[int(frame)]
        with np.load(source_root / row["path"], allow_pickle=False) as z:
            xyz = z["xyz"].astype(float)
        meta_path = person_root / "metadata" / f"{int(frame):06d}.json"
        meta = json.loads(meta_path.read_text())
        instances = {}
        for item in meta.get("instances", []):
            tid = mapping.get(str(item.get("sam_source_instance_id")))
            if tid is not None:
                instances[int(tid)] = item
        ix = np.flatnonzero(frames == frame)
        T = poses[frame]
        for i in ix:
            world_old = T[:3, :3] @ old[i] + T[:3, 3]
            world_final = np.r_[refs[i], world_old[2]]
            local_final[i] = T[:3, :3].T @ (world_final - T[:3, 3])
            item = instances.get(int(ids[i]), {})
            support = np.asarray(item.get("exclusive_support_points", []), dtype=int)
            support = support[(support >= 0) & (support < len(xyz))]
            cloud = xyz[support]
            valid = (np.isfinite(cloud).all(1) & (cloud >= [-20, -20, -3]).all(1)
                     & (cloud < [20, 20, 6]).all(1))
            support = support[valid]
            world = xyz[support] @ T[:3, :3].T + T[:3, 3]
            counts[i] = int((np.linalg.norm(world[:, :2] - refs[i], axis=1) <= .5).sum())
            match = item.get("yolo_match") or {}
            confirmed[i] = float(match.get("score", 0)) >= .4
            available[i] &= np.isfinite(refs[i]).all() and np.isfinite(local_final[i]).all()

    if "position_source" in data:
        data["ray_position_source"] = data["position_source"].copy()
    if "camera_depth" in data:
        data["ray_camera_depth"] = data.pop("camera_depth")
    data.update(corrected_point_lidar=local_final,
                corrected_xy_map=refs.copy(), reference_xy_map=refs.copy(),
                position_available=available,
                position_source=np.full(len(frames), "global_trajectory_reference_xy", dtype="<U36"),
                selected_support_counts=counts, track_only=counts < 5,
                class_confirmed=confirmed, position_policy=np.asarray(POLICY),
                ray_corrected_point_lidar=old)
    data["inputs_json"] = np.asarray(inputs)
    fd, tmp = tempfile.mkstemp(prefix=".final_centers.", suffix=".npz", dir=tracks)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **data)
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    digest = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    old_world = np.einsum("nij,nj->ni", poses[frames, :3, :3], old) + poses[frames, :3, 3]
    displacement = np.linalg.norm(refs - old_world[:, :2], axis=1)
    measurable = available & np.isfinite(displacement)
    report = dict(policy=POLICY, records=len(frames), available=int(available.sum()),
                  track_only=int((counts < 5).sum()),
                  max_xy_displacement_m=float(displacement[measurable].max(initial=0)),
                  z_policy="preserve old corrected_point_lidar transformed to map Z",
                  source_paths=[str(Path(p).resolve()) for p in paths],
                  input_sha256={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in paths if Path(p).is_file()}, diagnostic_sha256=digest)
    (tracks / "final_positions_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
