"""Prepare current-mask interior-half medians for SAM3 pedestrian trajectory fitting."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from person_mask_interior import select_interior_half


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=json.loads(args.config.read_text());root=Path(c['temporal_output']);video=root/'map_review/sam3_full_video';out=video/'interior_half_review';(out/'metadata').mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path(c['source_index']).read_text())['rows'];source=Path(c['source_index']).parent;person=Path(c['person_root']);records=[];presence={}
    with np.load(c['pose_npz']) as z:poses=z['transforms']
    if len(poses)!=len(rows):raise ValueError('Pose count differs from source frame count')
    cv2.setNumThreads(1)
    for row,T in zip(rows,poses):
        frame=int(Path(row['path']).stem);stem=f'{frame:06d}';assoc=json.loads((video/'associations'/f'{stem}.json').read_text())['observations'];items=[]
        with np.load(video/'masks'/f'{stem}.npz') as z:
            shape=tuple(z['mask_shape']);ids=z['object_ids'];masks=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)
        for tid,mask in zip(ids,masks):
            if mask.any():presence.setdefault(str(int(tid)),[]).append(frame)
        if assoc:
            with np.load(source/row['path']) as z:xyz=z['xyz'];uv=z['pixels_uv'];depth=z['camera_depth']
            with np.load(person/'frames'/f'{stem}.npz') as z:support=z['support_indices'];offset=z['support_offsets']
            for o in assoc:
                iid=o['instance_id'];tid=o['sam_video_id'];raw=support[offset[iid]:offset[iid+1]]
                r=dict(frame=frame,instance_id=iid,sam_video_id=tid,raw_xy=o['raw_xy'],raw_support_count=len(raw),selected_ids=[],status='unmatched_video_mask')
                if tid is not None:
                    selected,_,diag=select_interior_half(masks[np.flatnonzero(ids==tid)[0]],uv,raw);r.update(diag,selected_ids=selected.tolist(),status='review_candidate' if len(selected)>=3 else 'insufficient_interior_support')
                    if len(selected):
                        point=np.median(xyz[selected].astype(float),axis=0);world=T[:3,:3]@point+T[:3,3]
                        r.update(interior_xy=point[:2].tolist(),interior_xyz_map=world.tolist(),selected_depth_median=float(np.median(depth[selected])))
                        records.append(r)
                items.append(r)
        (out/'metadata'/f'{stem}.json').write_text(json.dumps(dict(frame=frame,observations=items),indent=2))
    np.savez_compressed(out/'map_centers_only.npz',xy_map=np.array([r['interior_xyz_map'][:2] for r in records]).reshape(-1,2),frame_ids=np.array([r['frame'] for r in records],int),instance_ids=np.array([r['instance_id'] for r in records],int),sam_video_ids=np.array([r['sam_video_id'] for r in records],int),selected_point_counts=np.array([r['selected_points'] for r in records],int),review_candidate=np.array([r['status']=='review_candidate' for r in records]))
    (video/'visual_presence_index.json').write_text(json.dumps(presence))
    # Dataset-specific manual category decisions are never inferred from track numbers.
    audit=video/'object_category_audit.json'
    if not audit.exists():audit.write_text(json.dumps(dict(excluded_objects=[],review_status='not_manually_reviewed')))
    report=dict(status='complete',frames=len(rows),observed_medians=len(records),temporal_accumulation=False,labels_modified=False)
    (out/'preparation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
