"""Apply validated multi-anchor refinements without changing retained vehicle identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from multi_anchor_vehicles import optimize_track


def load_observations(config):
    index=Path(config['source_index']);root=Path(config['temporal_output'])
    rows=json.loads(index.read_text())['rows'];groups={};digest=hashlib.sha256()
    with np.load(config['pose_npz'],allow_pickle=False) as saved:poses=saved['transforms']
    for row in rows:
        frame=int(row.get('local_frame_id',Path(row['path']).stem))
        path=root/'metadata'/f'{frame:06d}.json';blob=path.read_bytes()
        digest.update(path.name.encode());digest.update(blob)
        accepted=[o for o in json.loads(blob)['observations'] if o.get('kind')=='vehicle'
                  and o.get('track_id') is not None and o['status'] in ('accepted_temporal','accepted_single_frame')]
        if not accepted:continue
        with np.load(index.parent/row['path'],allow_pickle=False) as saved:
            xyz=saved['xyz'].astype(float);intrinsic=saved['cam_intrinsic'];extrinsic=saved['cam_extrinsic']
        for source in accepted:
            observation=dict(source,frame=frame,box=np.array(source['box'],float),
                             points=xyz[source['support_ids']],K=intrinsic,E=extrinsic,T=poses[frame])
            groups.setdefault(int(source['track_id']),[]).append(observation)
    for track in groups.values():
        if len({o['frame'] for o in track})!=len(track):raise ValueError('同一车辆轨迹在一帧中出现多条观测')
    return groups,digest.hexdigest()


def run(config_path):
    config=json.loads(Path(config_path).read_text());root=Path(config['temporal_output'])/'map_review/history_fit'
    history_path=root/'report.json';history_blob=history_path.read_bytes()
    baseline={int(r['track_id']):r for r in json.loads(history_blob)['results'] if 'candidate_box_map' in r}
    target=root/'bev_dedup/vehicle_map_boxes.npz'
    with np.load(target,allow_pickle=False) as saved:arrays={name:saved[name].copy() for name in saved.files}
    ids=arrays['track_ids'];previous=arrays['box_3d_map'].reshape(-1,7)
    groups,metadata_sha=load_observations(config)
    results=[]
    for tid,observations in sorted(groups.items()):
        if tid in baseline:
            result=optimize_track(tid,observations,np.array(baseline[tid]['candidate_box_map'],float))
        else:
            result=dict(track_id=tid,status='retain_original_no_global_reference',adopted=False,
                        source_frames=[int(o['frame']) for o in observations],source_observations=len(observations),
                        source_support_points=sum(len(o['points']) for o in observations),source_points_deleted=0,
                        final_box_map=None)
        results.append(result)
        print(json.dumps(dict(track_id=tid,status=result['status'],adopted=result['adopted'])),flush=True)
    by_id={r['track_id']:r for r in results}
    final=np.array([by_id[int(tid)]['final_box_map'] if int(tid) in by_id
                    and by_id[int(tid)]['final_box_map'] is not None else previous[i]
                    for i,tid in enumerate(ids)],float).reshape(-1,7)
    if not np.isfinite(final).all():raise ValueError('多锚帧优化生成了无效车辆框')
    np.testing.assert_array_equal(final[:,3:6],previous[:,3:6])
    changed=not np.array_equal(final,previous)
    out=root/'multi_anchor';out.mkdir(exist_ok=True)
    backup_key=hashlib.sha256(history_blob+ids.tobytes()).hexdigest()[:16]
    backup=out/'backups'/backup_key
    if changed:
        if not (backup/target.name).is_file():
            backup.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,backup/target.name);shutil.copy2(history_path,backup/'history_report.json')
        temporary=target.with_name(f'vehicle_map_boxes_{uuid4().hex}.tmp.npz')
        np.savez_compressed(temporary,**{**arrays,'box_3d_map':final})
        temporary.replace(target)
    adopted=[r for r in results if r['adopted']]
    retained=set(map(int,ids))
    report=dict(schema='lidar_multi_anchor_vehicles_v1',status='complete',
                algorithm='top8 reliable training anchors; fixed dimensions/camera ray; heldout validation gate',
                random_seed=20261018,validation_fraction=.2,maximum_anchors=8,maximum_yaw_change_deg=90,
                source_tracks=len(groups),source_observations=sum(len(v) for v in groups.values()),
                tracks_with_global_candidates=len(baseline),candidate_tracks_updated=len(adopted),
                retained_global_tracks=len(ids),retained_global_tracks_updated=sum(r['track_id'] in retained for r in adopted),
                retained_track_ids=ids.tolist(),tracks=results,source_metadata_sha256=metadata_sha,
                source_index_sha256=hashlib.sha256(Path(config['source_index']).read_bytes()).hexdigest(),
                pose_sha256=hashlib.sha256(Path(config['pose_npz']).read_bytes()).hexdigest(),
                global_identity_policy='preserve existing deduplicated identities and dimensions; no observation deletion',
                final_boxes=str(target),backup=str(backup) if backup.is_dir() else None,
                source_points_modified=False,person_results_modified=False)
    path=out/'report.json';payload=json.dumps(report,ensure_ascii=False,indent=2)
    if not path.is_file() or path.read_text()!=payload:
        temporary=path.with_name(f'report_{uuid4().hex}.tmp.json');temporary.write_text(payload);temporary.replace(path)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    report=run(args.config)
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
