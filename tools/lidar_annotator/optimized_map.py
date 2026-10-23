"""Accumulate every optimized task frame in the saved odometry map coordinates."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from .display_range import DEFAULT_LIMITS, point_mask
from .live_preview import frame_inputs, semantic_point_labels
from .recipe_config import RECIPE_PATH, load_recipe
from .track_preview import load_targets, ignored_support_indices, vehicle_boxes_in_frame
from .cli import REPO
from tools.autolabel.zgc_labeling.finalize_person_positions import load_positions


def transform_points(points, transform):
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def frame_number(row):
    return int(row.get('original_frame_id', row.get('source_frame_id', Path(row['path']).stem)))


def optimized_map_ready(context):
    out = Path(context['output']); backend = context.get('backend', {})
    temporal = Path(backend.get('temporal_output', out / 'temporal'))
    paths = (Path(backend.get('source_index', out / 'data/labels/zgc/index.json')),
             Path(backend.get('pose_npz', temporal / 'poses.npz')),
             temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz',
             temporal / 'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz')
    return all((p if p.is_absolute() else REPO / p).is_file() for p in paths)


def voxel_indices(xyz, labels, spacing):
    if spacing == 0 or not len(xyz):return np.arange(len(xyz))
    # Prefer labeled and actor points when several observations occupy one display voxel.
    priority = np.where(labels == 255, 0, np.where(np.isin(labels, [1, 2]), 2, 1))
    order = np.argsort(-priority, kind='stable')
    keys = np.floor(xyz[order] / spacing).astype(np.int32)
    _, first = np.unique(keys, axis=0, return_index=True)
    return order[first]


def frame_semantics(arrays, mask_path, targets, vehicle_boxes=None):
    mask = None
    if mask_path.is_file():
        with Image.open(mask_path) as image:mask = np.asarray(image)
    return semantic_point_labels(arrays, mask, targets, vehicle_boxes)


def _file_stamp(path):
    try:
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    except FileNotFoundError:return [str(path.resolve()), None]


def _input_stamp(path):
    if path.suffix in ('.json','.py') and path.is_file():
        return [str(path.resolve()),hashlib.sha256(path.read_bytes()).hexdigest()]
    return _file_stamp(path)


def _map_signature(context):
    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else REPO / path
    out = absolute(context['output']); backend = context.get('backend', {})
    index = absolute(backend.get('source_index', out / 'data/labels/zgc/index.json'))
    temporal = absolute(backend.get('temporal_output', out / 'temporal'))
    video = temporal / 'map_review/sam3_full_video'
    tracks = video / 'interior_half_review/consistent_tracks'
    people = absolute(backend.get('person_root', out / 'data/person'))
    inputs = [index, out / 'raw/index.json', absolute(backend.get('pose_npz', temporal / 'poses.npz')),
              tracks / 'centers_only.npz', tracks / 'final_centers.npz', tracks / 'report.json', video / 'native_to_display_ids.json',
              temporal / 'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz', out / 'recipe.json', RECIPE_PATH]
    for row in json.loads(index.read_text())['rows']:
        stem = Path(row['path']).stem
        inputs += [index.parent / row['path'], index.parent / 'image_masks' / f'{stem}.png',
                   out / 'data/visual_instances' / f'{stem}.json', out / 'data/visual_instances' / f'{stem}.npz',
                   video / 'masks' / f'{stem}.npz', people / 'metadata' / f'{stem}.json']
        inputs += sorted((tracks/'metadata').glob(f'{stem}_I*.json'))
    inputs += [Path(__file__).with_name(name) for name in
               ('optimized_map.py', 'live_preview.py', 'track_preview.py', 'person_support.py', 'display_range.py', 'recipe_config.py')]
    inputs.append(REPO/'tools/autolabel/zgc_labeling/finalize_person_positions.py')
    values = [2, list(context.get('map_limits', DEFAULT_LIMITS)), [_input_stamp(path) for path in inputs]]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


_DISPLAY_ARRAYS = ('xyz', 'labels', 'person_xyz', 'person_ids', 'person_frames', 'person_segments', 'person_track_only',
                   'boxes', 'car_ids', 'trajectory', 'trajectory_frames', 'segments')


def _write_map_report(data):
    manifest = dict(frame_count=data['frame_count'], point_count=data['point_count'],
                    display_points=len(data['xyz']), display_voxel_m=data['spacing'],
                    original_frame_ids=data['trajectory_frames'].tolist(),
                    annotation='saved optimized person positions and deduplicated vehicle boxes',
                    coordinates='saved recorded-odometry transforms; no additional point-cloud registration',
                    per_frame_limits=list(data['limits']), scope='all processed frames of the current task',
                    ignored_support_points=data['ignored_count'], labels_modified=False,
                    person_position_policy=data.get('person_position_policy','legacy_camera_ray'),
                    trajectory_only_positions=int(np.asarray(data.get('person_track_only',[])).sum()))
    path = Path(data['directory']) / 'report.json'
    temporary = path.with_name(f'report_{uuid4().hex}.tmp.json')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)); temporary.replace(path)


def _read_map_archive(archive, spacing):
    with np.load(archive, allow_pickle=False) as z:
        xyz, labels = z['xyz_map'], z['point_raw_labels']
        selected = voxel_indices(xyz, labels, spacing)
        return dict(xyz=xyz[selected], labels=labels[selected], person_xyz=z['person_xyz_map'],
                    person_ids=z['person_ids'], person_frames=z['person_frame_ids'], person_segments=z['person_segments'],person_track_only=z['person_track_only'],
                    boxes=z['vehicle_boxes_map'], car_ids=z['vehicle_ids'], trajectory=z['poses'][:, :3, 3].astype(np.float32),
                    trajectory_frames=z['pose_frame_ids'], segments=z['pose_segments'])


def load_optimized_map(context, spacing=.15, progress=None, cancelled=None):
    """Reuse a saved display/HTML, or resample the full archive without rereading frames."""
    if not np.isfinite(spacing) or spacing < 0:raise ValueError('显示体素间距必须大于等于 0')
    def check_cancelled():
        if cancelled and cancelled():raise InterruptedError('地图加载已取消')
    check_cancelled()
    signature = _map_signature(context)
    out = Path(context['output']); out = out if out.is_absolute() else REPO / out
    destination = out / 'review/optimized_map'
    archive, display, html, receipt = [destination / name for name in
                                      ('map_points.npz', 'display_points.npz', 'map_3d.html', 'cache.json')]
    cached = None; data = None; display_reused = False
    try:
        saved = json.loads(receipt.read_text())
        if saved['signature'] == signature and archive.is_file() and saved['archive'] == _file_stamp(archive):
            metadata = {name: saved['metadata'][name] for name in ('frame_count', 'point_count', 'ignored_count', 'palette','person_position_policy')}
            cached = saved
            if saved['spacing'] == spacing and display.is_file() and saved['display'] == _file_stamp(display):
                with np.load(display, allow_pickle=False) as z:data = {name: z[name] for name in _DISPLAY_ARRAYS}
                display_reused = True
            else:data = _read_map_archive(archive, spacing)
            data.update(metadata)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        cached = None; data = None; display_reused = False
    check_cancelled()
    if data is None:
        data = build_optimized_map(context, spacing, progress, cancelled)
    data.update(spacing=spacing, limits=tuple(context.get('map_limits', DEFAULT_LIMITS)),
                archive=str(archive), directory=str(destination), cache_reused=cached is not None)
    check_cancelled()
    if display_reused and html.is_file() and cached.get('html') == _file_stamp(html):
        data['html'] = str(html)
        return data
    # Publish the receipt last so cancelled or interrupted writes cannot be reused.
    if not display_reused:
        temporary = destination / f'display_{uuid4().hex}.tmp.npz'
        np.savez_compressed(temporary, **{name: data[name] for name in _DISPLAY_ARRAYS})
        temporary.replace(display)
        if cached is not None:_write_map_report(data)
    check_cancelled()
    data['html'] = str(export_map_html(data))
    check_cancelled()
    metadata = {name: data[name] for name in ('frame_count', 'point_count', 'ignored_count', 'palette','person_position_policy')}
    saved = dict(signature=signature, spacing=spacing, metadata=metadata,
                 archive=_file_stamp(archive), display=_file_stamp(display), html=_file_stamp(html))
    temporary = receipt.with_name(f'cache_{uuid4().hex}.tmp.json')
    temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2)); temporary.replace(receipt)
    return data


def build_optimized_map(context, spacing=.15, progress=None, cancelled=None):
    if not np.isfinite(spacing) or spacing < 0:raise ValueError('显示体素间距必须大于等于 0')
    def absolute(value):
        path = Path(value)
        return path if path.is_absolute() else REPO / path
    out = absolute(context['output']); backend = context.get('backend', {})
    source_index = Path(backend.get('source_index', out / 'data/labels/zgc/index.json'))
    temporal = Path(backend.get('temporal_output', out / 'temporal'))
    poses_path = Path(backend.get('pose_npz', temporal / 'poses.npz'))
    persons_path = temporal / 'map_review/sam3_full_video/interior_half_review/consistent_tracks/centers_only.npz'
    vehicles_path = temporal / 'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz'
    source_index, poses_path, temporal = map(absolute, (source_index, poses_path, temporal))
    persons_path, vehicles_path = map(absolute, (persons_path, vehicles_path))
    for path in (source_index, poses_path, persons_path, vehicles_path):
        if not path.is_file():raise ValueError('请先完成全部帧的语义、实例与时序优化；缺少：' + str(path))
    rows = json.loads(source_index.read_text())['rows']
    if not rows:raise ValueError('没有可用于地图的帧')
    limits = tuple(context.get('map_limits', DEFAULT_LIMITS))
    with np.load(poses_path, allow_pickle=False) as z:
        poses = z['transforms'].copy()
        segments = z['segments'].copy() if 'segments' in z else np.zeros(len(poses), int)
    locals_ = [int(row.get('local_frame_id', Path(row['path']).stem)) for row in rows]
    if len(set(locals_)) != len(rows) or min(locals_) < 0 or max(locals_) >= len(poses):
        raise ValueError('标签帧与保存位姿不匹配，不能累积地图')
    if poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():raise ValueError('地图位姿无效')
    people = load_positions(persons_path)
    with np.load(vehicles_path, allow_pickle=False) as z:boxes = z['box_3d_map'].reshape(-1, 7).copy(); car_ids = z['track_ids'].copy()
    good = people['position_available'] & np.isfinite(people['corrected_point_lidar']).all(1)
    good &= np.isin(people['frame_ids'], locals_) & point_mask(people['corrected_point_lidar'], limits)
    person_frames = people['frame_ids'][good].astype(int)
    person_xyz = np.empty((int(good.sum()), 3), np.float32)
    for i, (point, local) in enumerate(zip(people['corrected_point_lidar'][good], person_frames)):
        person_xyz[i] = transform_points(point[None], poses[local])[0]
    good_boxes = np.isfinite(boxes).all(1) & (boxes[:, 3:6] > 0).all(1)
    boxes, car_ids = boxes[good_boxes], car_ids[good_boxes]
    visible_boxes = np.array([any(box_visible(box, poses[local], limits) for local in locals_) for box in boxes], bool)
    boxes, car_ids = boxes[visible_boxes], car_ids[visible_boxes]
    resolved = dict(context, output=str(out), backend={**backend, 'source_index': str(source_index), 'pose_npz': str(poses_path), 'temporal_output': str(temporal)})
    base_paths = frame_inputs(resolved, frame_number(rows[0]), True)
    recipe_path = out / 'recipe.json'
    recipe = load_recipe(recipe_path if recipe_path.is_file() else None)
    mapping_path = base_paths['id_map']
    mapping = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    xyz_chunks, label_chunks, frame_chunks = [], [], []
    ignored = 0
    for ordinal, (row, local) in enumerate(zip(rows, locals_)):
        if cancelled and cancelled():raise InterruptedError('地图加载已取消')
        stem = Path(row['path']).stem
        with np.load(source_index.parent / row['path'], allow_pickle=False) as z:arrays = {k: z[k] for k in z.files}
        paths = dict(base_paths, local=local, stem=stem,
                     mask=source_index.parent / 'image_masks' / f'{stem}.png',
                     person_metadata=base_paths['person_metadata'].parent / f'{stem}.json',
                     instance_masks=out / 'data/visual_instances' / f'{stem}.npz',
                     track_masks=temporal / 'map_review/sam3_full_video/masks' / f'{stem}.npz')
        records_path = out / 'data/visual_instances' / f'{stem}.json'
        records = json.loads(records_path.read_text()).get('records', []) if records_path.is_file() else []
        for record in records:record['display_id'] = mapping.get(str(record.get('track_id')), record.get('track_id'))
        targets = load_targets(paths, arrays, records, limits)
        rejected = ignored_support_indices(targets)
        valid = point_mask(arrays['xyz'], limits)
        ignored += int(valid[rejected].sum())
        local_boxes = vehicle_boxes_in_frame(boxes, poses[local])
        labels = frame_semantics(arrays, source_index.parent / 'image_masks' / f'{stem}.png', targets, local_boxes)
        xyz_chunks.append(transform_points(arrays['xyz'][valid], poses[local]))
        label_chunks.append(labels[valid]); frame_chunks.append(np.full(int(valid.sum()), frame_number(row), np.int32))
        if progress:progress(ordinal + 1, len(rows))
    xyz = np.concatenate(xyz_chunks); labels = np.concatenate(label_chunks); frames = np.concatenate(frame_chunks)
    if cancelled and cancelled():raise InterruptedError('地图加载已取消')
    key = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]
    original = {local: frame_number(row) for row, local in zip(rows, locals_)}
    person_original_frames = np.array([original[f] for f in person_frames], np.int32)
    map_segments = np.cumsum(np.r_[False, (np.diff([original[f] for f in locals_]) != 1) | (np.diff(segments[locals_]) != 0)])
    segment_by_local = dict(zip(locals_, map_segments))
    person_segments = np.asarray([segment_by_local[f] for f in person_frames], np.int32)
    destination = out / 'review/optimized_map'
    destination.mkdir(parents=True, exist_ok=True)
    # Keep every accumulated point in the archive; thinning applies only to display.
    temporary = destination / f'map_points_{key}.tmp.npz'
    np.savez_compressed(temporary, xyz_map=xyz, point_raw_labels=labels, source_frame_ids=frames,
                        person_xyz_map=person_xyz, person_ids=people['sam_video_ids'][good],
                        person_track_only=people['track_only'][good],
                        person_frame_ids=person_original_frames, person_segments=person_segments, vehicle_boxes_map=boxes, vehicle_ids=car_ids,
                        poses=poses[locals_], pose_frame_ids=[original[f] for f in locals_], pose_segments=map_segments)
    archive = destination / 'map_points.npz'; temporary.replace(archive)
    selected = voxel_indices(xyz, labels, spacing)
    result = dict(xyz=xyz[selected], labels=labels[selected], frame_count=len(rows), point_count=len(xyz),
                  ignored_count=ignored, spacing=spacing, person_xyz=person_xyz,
                  person_ids=people['sam_video_ids'][good], person_frames=person_original_frames, person_segments=person_segments,
                  person_track_only=people['track_only'][good],
                  person_position_policy=people['position_policy'],
                  boxes=boxes, car_ids=car_ids, trajectory=poses[locals_, :3, 3].astype(np.float32),
                  trajectory_frames=np.asarray([original[f] for f in locals_], np.int32),
                  segments=map_segments, palette=recipe['raw_classes'], archive=str(archive),
                  directory=str(destination), limits=limits)
    _write_map_report(result)
    return result


def box_corners(box):
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * box[3:5] / 2
    co, si = np.cos(box[6]), np.sin(box[6])
    xy = xy @ np.array([[co, si], [-si, co]]) + box[:2]
    return np.concatenate([np.c_[xy, np.full(4, box[2] - box[5] / 2)],
                           np.c_[xy, np.full(4, box[2] + box[5] / 2)]])


def box_visible(box, transform, limits):
    inverse = np.linalg.inv(transform); local = box.copy()
    local[:3] = inverse[:3, :3] @ box[:3] + inverse[:3, 3]
    local[6] += np.arctan2(inverse[1, 0], inverse[0, 0])
    xy = box_corners(local)[:4, :2]; extent, zmin, zmax = limits
    return not ((xy.max(0) < -extent).any() or (xy.min(0) > extent).any() or local[2] + local[5] / 2 < zmin or local[2] - local[5] / 2 > zmax)


def person_trajectory_segments(data):
    """Connect saved centres only within one identity and consecutive observed frames."""
    for tid in np.unique(data['person_ids']):
        indices = np.flatnonzero(data['person_ids'] == tid)
        indices = indices[np.argsort(data['person_frames'][indices], kind='stable')]
        breaks = (np.diff(data['person_frames'][indices]) != 1) | (np.diff(data['person_segments'][indices]) != 0)
        cuts = np.r_[0, np.flatnonzero(breaks) + 1, len(indices)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end - start >= 2:yield tid, indices[start:end]


def person_track_starts(data):
    for tid in np.unique(data['person_ids']):
        indices = np.flatnonzero(data['person_ids'] == tid)
        yield tid, indices[np.argmin(data['person_frames'][indices])]


def export_map_html(data):
    import plotly.graph_objects as go
    palette = {c['id']: c for c in data['palette']}
    figure = go.Figure()
    for cid in [255, *range(10)]:
        points = data['xyz'][data['labels'] == cid]
        if not len(points):continue
        item = palette.get(cid, dict(name='未知 / 背景', rgb=[110, 110, 110])); rgb = item['rgb']
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
            name=item['name'], marker=dict(size=1.5, color=f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'), hoverinfo='skip'))
    people = data['person_xyz']
    track_only=np.asarray(data.get('person_track_only',np.zeros(len(people),bool)))
    for selected,name,color,symbol in ((~track_only,'行人位置（有当前点支持）','#00ffff','circle'),(track_only,'行人位置（仅轨迹估计）','#ffbf66','x')):
        if not selected.any():continue
        points=people[selected]
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', name=name,
            customdata=np.c_[data['person_ids'][selected], data['person_frames'][selected]], marker=dict(size=4, color=color,symbol=symbol),
            hovertemplate='行人 P%{customdata[0]} · 帧 %{customdata[1]}<br>X %{x:.2f} Y %{y:.2f} Z %{z:.2f}<extra></extra>'))
    shown = set()
    for tid, indices in person_trajectory_segments(data):
        points = people[indices]
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines',
            name=f'行人 P{tid} 中心轨迹', legendgroup=f'person_{tid}', showlegend=tid not in shown,
            line=dict(color='#00ffff', width=4)))
        shown.add(tid)
    for tid, index in person_track_starts(data):
        point = people[index]
        figure.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers',
            name=f'行人 P{tid} 起点', legendgroup=f'person_{tid}', marker=dict(size=7, color='#00ffff', symbol='diamond'),
            customdata=[int(data['person_frames'][index])], hovertemplate=f'行人 P{tid} 起点 · 帧 %{{customdata}}<extra></extra>'))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), *[(i, i + 4) for i in range(4)]]
    for box, tid in zip(data['boxes'], data['car_ids']):
        corners = box_corners(box); points = np.array([p for a, b in edges for p in (corners[a], corners[b], [np.nan] * 3)])
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines', name=f'车辆 V{tid}', line=dict(color='#ffdc00', width=4)))
    for segment in np.unique(data['segments']):
        path = data['trajectory'][data['segments'] == segment]
        figure.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', name=f'采集轨迹 {segment}', line=dict(color='white', width=3)))
    path = data['trajectory']
    figure.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='markers', name='每帧采集位置',
        customdata=data['trajectory_frames'], marker=dict(size=3, color='white'),
        hovertemplate='帧 %{customdata}<br>X %{x:.2f} Y %{y:.2f} Z %{z:.2f}<extra></extra>'))
    figure.add_trace(go.Scatter3d(x=path[:1, 0], y=path[:1, 1], z=path[:1, 2], mode='markers', name='起始帧',
        customdata=data['trajectory_frames'][:1], marker=dict(size=6, color='white', symbol='diamond'),
        hovertemplate='起始帧 %{customdata}<br>X %{x:.2f} Y %{y:.2f} Z %{z:.2f}<extra></extra>'))
    figure.update_layout(template='plotly_dark', title=f"全部 {data['frame_count']} 帧 · 优化后目标标注地图 · 显示体素 {data['spacing']:g} m",
        scene=dict(aspectmode='data', xaxis_title='Map X (m)', yaxis_title='Map Y (m)', zaxis_title='Map Z (m)'), margin=dict(l=0, r=0, b=0, t=45))
    path = Path(data['directory']) / 'map_3d.html'; figure.write_html(path, include_plotlyjs=True)
    return path
