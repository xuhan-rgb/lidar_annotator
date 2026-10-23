#!/usr/bin/env python3
"""Validate every exported pedestrian frame and write full-data accounting."""
import json
import sys
from pathlib import Path

import numpy as np

from config import OUT,SOURCE,OLD_SUPERVISION,TRAINING_ROI,sha256
from geometry import box_to_corners


def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else OUT
    source=json.loads((SOURCE/'index.json').read_text())['rows'];source_by={Path(r['path']).stem:r for r in source}
    index=json.loads((root/'index.json').read_text());records=index['frames'];assert len(source)==971 and len(records)==971
    assert index['schema']=='pedestrian_instance_labels_v3'
    assert {r['frame'] for r in records}==set(source_by)
    counts=dict(frames=0,xy_instances=0,box_3d_instances=0,rejected_instances=0,
        unmatched_wilddet=0,unmatched_yolo=0,legacy_negative_cells=0,effective_negative_cells=0,
        ignored_cells=0,positive_negative_conflicts=0,unique_support_points=0)
    support_sizes=[];confidence=[];sourceid_hash=sha256(SOURCE/'index.json')
    assert index['source_index_sha256']==sourceid_hash
    for record in records:
        stem=record['frame'];row=source_by[stem];npz=root/record['npz'];meta_path=root/record['metadata']
        assert sha256(npz)==record['npz_sha256'] and sha256(meta_path)==record['metadata_sha256']
        meta=json.loads(meta_path.read_text())
        assert meta['frame']==int(stem) and meta['frame_id']==row['frame_id'] and meta['source_frame_id']==row['frame_id'] and meta['source_frame_number']==int(stem)
        assert meta['manual_ground_truth'] is False and meta['added_to_training'] is False
        with np.load(npz) as z:
            assert z['frame_id'].item()==row['frame_id'] and int(z['source_frame_id'])==int(stem)
            assert z['xy'].ndim==2 and z['xy'].shape[1]==2 and z['box_3d'].ndim==2 and z['box_3d'].shape[1]==7
            assert z['confidence'].shape==(len(z['xy']),) and z['box_3d_confidence'].shape==(len(z['box_3d']),)
            assert z['legacy_negative_mask'].shape==(80,80) and z['ignore_mask'].shape==(80,80) and z['effective_negative_mask'].shape==(80,80)
            assert np.isfinite(z['xy']).all() and np.isfinite(z['box_3d']).all() and np.isfinite(z['confidence']).all()
            assert np.all((z['confidence']>=0)&(z['confidence']<=1))
            assert not np.any(z['effective_negative_mask']&z['ignore_mask'])
            assert np.array_equal(z['effective_negative_mask'],z['legacy_negative_mask']&~z['ignore_mask'])
            for xy in z['xy']:
                cell=np.floor((xy-np.asarray(TRAINING_ROI[:2]))/.5).astype(int)
                if np.all((cell>=0)&(cell<80)):
                    assert z['ignore_mask'][cell[1],cell[0]] and not z['effective_negative_mask'][cell[1],cell[0]]
            if len(z['xy']):assert np.all((z['xy']>=-20)&(z['xy']<20))
            if len(z['box_3d']):
                assert np.isfinite(z['box_3d']).all() and np.all((z['box_3d'][:,:3]>=np.asarray(TRAINING_ROI[:3]))&(z['box_3d'][:,:3]<np.asarray(TRAINING_ROI[3:])))
                assert z['box_3d_corners_lidar'].shape==(len(z['box_3d']),8,3)
                for box,corners in zip(z['box_3d'],z['box_3d_corners_lidar']):
                    assert np.allclose(corners,box_to_corners(box),atol=1e-6),f'{stem}: stored 3D corners disagree with float32 box'
            else:assert z['box_3d_corners_lidar'].shape==(0,8,3)
            assert z['box_3d'].shape==(0,7) and z['box_3d_instance_ids'].shape==(0,)
            ids=np.asarray(z['support_indices']);offset=np.asarray(z['support_offsets']);assert offset.shape==(len(meta['instances'])+1,)
            owned=[]
            for a,b in zip(offset[:-1],offset[1:]):
                owned.extend(ids[int(a):int(b)].tolist());support_sizes.append(int(b-a))
            assert len(owned)==len(set(owned)),f'{stem}: LiDAR point reused across SAM instances'
            assert z['support_xyz'].shape==(len(ids),3)
            counts['unique_support_points']+=len(owned)
            counts['xy_instances']+=len(z['xy']);counts['box_3d_instances']+=len(z['box_3d'])
            counts['legacy_negative_cells']+=int(z['legacy_negative_mask'].sum());counts['effective_negative_cells']+=int(z['effective_negative_mask'].sum())
            counts['ignored_cells']+=int(z['ignore_mask'].sum());counts['positive_negative_conflicts']+=int(np.count_nonzero(z['effective_negative_mask']&z['ignore_mask']))
            confidence.extend(z['confidence'].tolist())
        assert record['xy_instances']==len(meta['accepted'])
        assert record['box_3d_instances']==0 and not any(bool(x.get('box_3d_valid')) for x in meta['accepted'])
        assert all(x['manual_ground_truth'] is False and x['box_3d_valid'] is False and x['box_3d'] is None
                   and x['person_class_confirmation_sources']==['matched_yolo26x_person_box'] and x['yolo_detection_id'] is not None
                   and x['supervision_mask']==dict(x=True,y=True,z=False,length=False,width=False,height=False,yaw=False)
                   and x.get('reasons')==[] for x in meta['accepted'])
        accepted_ids={int(x['instance_id']) for x in meta['accepted']}
        for inst in meta['instances']:
            confirmed=inst.get('yolo_match') is not None
            if not confirmed:assert int(inst['instance_id']) not in accepted_ids and 'unconfirmed_by_yolo26x_person' in inst.get('reasons',[])
        with np.load(npz) as z:
            for item in meta['accepted']:
                instance=meta['instances'][int(item['instance_id'])]
                a,b=int(instance['support_offset_start']),int(instance['support_offset_end'])
                point_ids=z['support_indices'][a:b]
                assert len(point_ids)==int(item['points'])>=5
                median=np.median(z['support_xyz'][a:b,:2].astype(float),axis=0)
                assert np.allclose(median,np.asarray(item['xy']),atol=2e-6),f'{stem} instance {item["instance_id"]}: XY is not median of every saved support point'
        with np.load(npz) as z:
            for inst in meta['instances']:
                a,b=inst['support_offset_start'],inst['support_offset_end']
                if b-a>=5:
                    center=np.median(z['support_xyz'][a:b,:2].astype(float),axis=0)
                    cell=np.floor((center-np.asarray(TRAINING_ROI[:2]))/.5).astype(int)
                    if np.all((cell>=0)&(cell<80)):
                        assert z['ignore_mask'][cell[1],cell[0]] and not z['effective_negative_mask'][cell[1],cell[0]]
        counts['rejected_instances']+=len(meta['rejected']);counts['unmatched_wilddet']+=len(meta['wilddet_unmatched']);counts['unmatched_yolo']+=len(meta['yolo_unmatched'])
        counts['frames']+=1
    counts.update(added_to_training=False,manual_ground_truth=False,
        confidence_full_export_min=float(min(confidence)) if confidence else None,
        confidence_full_export_median=float(np.median(confidence)) if confidence else None,
        support_points_per_sam_instance_full_export_min=min(support_sizes) if support_sizes else None,
        support_points_per_sam_instance_full_export_median=float(np.median(support_sizes)) if support_sizes else None,
        source_index_sha256=sourceid_hash,sha256_checked=True,random_sampling=False,summary_scope='all 971 frames')
    (root/'validation.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
