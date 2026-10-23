#!/usr/bin/env python3
"""CPU association, camera-ray depth refinement, and XY-only pseudo-label export."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import time
import hashlib

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from config import (DATA,OUT,SOURCE,YOLO,OLD_SUPERVISION,TRAINING_ROI,NEGATIVE_GRID_SHAPE,
                    NEGATIVE_GRID_RESOLUTION_M,YOLO_PERSON_SCORE_MIN,WILDDET_PROMPT,
                    WILDDET_COMBINED_SCORE_MIN,WILDDET_SCORE_3D_MIN,MIN_SUPPORT_POINTS,
                    SAM3_PROMPT,SAM3_INSTANCE_SCORE_MIN,SAM3_MASK_THRESHOLD,choose_smoke_frames,sha256,
                    WILDDET_WEIGHTS,WILDDET_WEIGHT_MANIFEST,SAM3_WEIGHTS)
from association import associate_boxes_to_masks,support_point_indices,mask_bbox,box_iou


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(obj):
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        raise TypeError(f'Cannot JSON encode {type(obj).__name__}')
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=convert));tmp.replace(path)


def load_sam_frame(frame,sam_root):
    stem=f'{frame:06d}';meta=json.loads((sam_root/'metadata'/f'{stem}.json').read_text())
    path=sam_root/'frames'/f'{stem}.npz'
    if sha256(path)!=meta['mask_sha256']:raise ValueError(f'SAM mask checksum mismatch in {stem}')
    with np.load(path) as z:
        shape=tuple(map(int,z['mask_shape'].tolist()));packed=z['masks_packed'].copy()
        masks=np.unpackbits(packed,axis=1,count=shape[0]*shape[1],bitorder='little').reshape(-1,*shape).astype(bool)
    if len(masks)!=len(meta['candidates']):raise ValueError(f'SAM instance count mismatch in {stem}')
    meta['instances']=meta['candidates']
    return masks,meta


def load_wild_frame(frame,wild_root):
    return json.loads((wild_root/'frames'/f'{frame:06d}.json').read_text())


def load_yolo_frame(frame):
    return json.loads((YOLO/'metadata'/f'{frame:06d}.json').read_text())


def summarize_support(indices,xyz):
    indices=np.asarray(indices,dtype=np.int64)
    if len(indices)<MIN_SUPPORT_POINTS:return dict(accepted=False,reason='fewer_than_5_exclusive_mask_lidar_points',indices=indices,support_count=int(len(indices)))
    xy=xyz[indices,:2];center=np.median(xy,axis=0)
    spread=float(np.median(np.linalg.norm(xy-center,axis=1)))
    return dict(accepted=True,indices=indices,center_xy=center.astype(float).tolist(),support_count=int(len(indices)),
        xy_spread_m=spread,xy_min_m=xy.min(axis=0).tolist(),xy_max_m=xy.max(axis=0).tolist(),
        position_method='coordinatewise_median_of_all_exclusive_valid_roi_points_no_clustering')


def _grid_cells(xy):
    xy=np.asarray(xy,float)
    origin=np.asarray(TRAINING_ROI[:2]);resolution=float(NEGATIVE_GRID_RESOLUTION_M)
    cells=np.floor((xy-origin)/resolution).astype(int)
    valid=np.isfinite(xy).all(1)&(cells[:,0]>=0)&(cells[:,0]<NEGATIVE_GRID_SHAPE[1])&(cells[:,1]>=0)&(cells[:,1]<NEGATIVE_GRID_SHAPE[0])
    return cells,valid


def ignore_grid_from_points(point_indices,xyz,candidate_xy=()):
    mask=np.zeros(NEGATIVE_GRID_SHAPE,bool)
    if len(point_indices):
        cells,valid=_grid_cells(xyz[np.asarray(point_indices,dtype=np.int64),:2])
        cells=cells[valid]
        if len(cells):mask[cells[:,1],cells[:,0]]=True
    if len(candidate_xy):
        cells,valid=_grid_cells(candidate_xy);cells=cells[valid]
        if len(cells):mask[cells[:,1],cells[:,0]]=True
    return mask


def points_in_boxes(boxes,uv,valid):
    ignored=[]
    for box in boxes:
        b=np.asarray(box,float)
        if b.shape!=(4,) or not np.isfinite(b).all():continue
        selected=valid&(uv[:,0]>=b[0])&(uv[:,0]<b[2])&(uv[:,1]>=b[1])&(uv[:,1]<b[3])
        ignored.extend(np.flatnonzero(selected).tolist())
    return np.unique(np.asarray(ignored,dtype=np.int64)) if ignored else np.empty(0,np.int64)


def _wild_passes(item):
    return (float(item.get('score',0))>=WILDDET_COMBINED_SCORE_MIN
            and float(item.get('score_3d',0))>=WILDDET_SCORE_3D_MIN)


def _xy_confidence(mask_score,support_count,spread):
    support_reliability=min(1.,float(support_count)/10.)*np.exp(-float(spread)/.4)
    return float(np.clip(float(mask_score)*support_reliability,0.,1.))


def person_instance_confirmation(sam_score,yolo_matched,wilddet_matched):
    """Only a one-to-one YOLO26x person match confirms a positive label."""
    sources=['matched_yolo26x_person_box'] if yolo_matched else []
    return bool(sources),sources


def process_frame(frame,source_row,sam_root,wild_root,output_root,render=False):
    stem=f'{frame:06d}';sam_masks,sam_meta=load_sam_frame(frame,sam_root)
    wild=load_wild_frame(frame,wild_root);yolo=load_yolo_frame(frame)
    if wild.get('frame_id')!=source_row['frame_id'] or sam_meta.get('frame_id')!=source_row['frame_id']:
        raise ValueError(f'frame_id mismatch at {stem}')
    persons=[d for d in yolo.get('detections',[]) if float(d.get('score',0))>=YOLO_PERSON_SCORE_MIN]
    raw_wild=wild.get('items',[])
    eligible_wild=[d for d in raw_wild if _wild_passes(d)]
    wild_assoc,wild_metrics,wild_eligible=associate_boxes_to_masks(eligible_wild,sam_masks,sam_meta['instances'])
    yolo_assoc,yolo_metrics,yolo_eligible=associate_boxes_to_masks(persons,sam_masks,sam_meta['instances'])
    wild_by_inst={int(a['instance_id']):eligible_wild[j] for j,a in enumerate(wild_assoc) if a['status']=='matched'}
    wild_assoc_by_inst={int(a['instance_id']):a for a in wild_assoc if a['status']=='matched'}
    yolo_by_inst={int(a['instance_id']):persons[j] for j,a in enumerate(yolo_assoc) if a['status']=='matched'}
    yolo_assoc_by_inst={int(a['instance_id']):a for a in yolo_assoc if a['status']=='matched'}
    with np.load(SOURCE/'frames'/f'{stem}.npz') as archive:
        xyz=np.asarray(archive['xyz'],float);uv=np.asarray(archive['pixels_uv'],float)
        visible=np.asarray(archive['visible'],bool);depth=np.asarray(archive['camera_depth'],float)
        labels=np.asarray(archive['point_labels'],np.uint8);E=np.asarray(archive['cam_extrinsic'],float)
        timestamp=int(np.asarray(archive['lidar_timestamp_ns']).item())
    per_instance,point_owner,cover,valid_points=support_point_indices(xyz,uv,visible,depth,sam_masks,TRAINING_ROI)
    yolo_points=points_in_boxes([d['bbox_xyxy'] for d in persons],uv,valid_points)
    wild_points=points_in_boxes([d['bbox_xyxy'] for d in raw_wild],uv,valid_points)
    all_sam_mask_points=[]
    for m in sam_masks:
        ys,xs=np.nonzero(m)
        if not len(xs):continue
        pp=np.floor(np.where(np.isfinite(uv),uv,0)).astype(int)
        sel=valid_points&(pp[:,0]>=0)&(pp[:,0]<m.shape[1])&(pp[:,1]>=0)&(pp[:,1]<m.shape[0])
        ids=np.flatnonzero(sel);all_sam_mask_points.extend(ids[m[pp[ids,1],pp[ids,0]]].tolist())
    ignored_ids=np.unique(np.concatenate([np.asarray(all_sam_mask_points,np.int64),yolo_points,wild_points]))
    ignore_mask=ignore_grid_from_points(ignored_ids,xyz)
    old_neg_path=OLD_SUPERVISION/'frames'/f'{stem}.npz'
    with np.load(old_neg_path) as old:negative=old['negative_mask'].astype(bool).copy()
    if negative.shape!=NEGATIVE_GRID_SHAPE:raise ValueError(f'old negative grid mismatch in {stem}')
    accepted=[];rejected=[];support_offsets=[0];support_indices=[];support_point_cloud=[];box3d=[];box3d_corners=[];box3d_ids=[];box3d_conf=[]
    instance_records=[];xy_records=[];candidate_xy=[]
    for iid,base in enumerate(sam_meta['instances']):
        assigned=per_instance[iid]
        support=summarize_support(assigned,xyz)
        if support['accepted']:candidate_xy.append(support['center_xy'])
        support['legacy_semantic_person_support_count']=int(np.count_nonzero(labels[assigned]==1)) if len(assigned) else 0
        support['raw_exclusive_mask_point_count']=int(len(assigned))
        wild_item=wild_by_inst.get(iid);wild_link=wild_assoc_by_inst.get(iid)
        yolo_item=yolo_by_inst.get(iid);yolo_link=yolo_assoc_by_inst.get(iid)
        person_confirmed,confirmation_sources=person_instance_confirmation(float(base['score']),yolo_item is not None,wild_item is not None)
        instance=dict(instance_id=int(iid),sam_source_instance_id=int(base['source_instance_id']),
            sam_prompt=base['prompt'],sam_score=float(base['score']),sam_bbox_xyxy=base['bbox_xyxy'],
            sam_area_pixels=int(base['area_pixels']),wilddet_match=None,yolo_match=None,
            wilddet_instance_id=None,yolo_detection_id=None,point_support_count=int(len(assigned)),
            person_class_confirmed=person_confirmed,person_class_confirmation_sources=confirmation_sources,
            legacy_semantic_person_support_count=support['legacy_semantic_person_support_count'],
            exclusive_support_points=np.asarray(assigned,np.int64).tolist())
        if wild_item is not None:
            instance['wilddet_instance_id']=int(wild_item.get('wilddet_instance_id',-1))
            instance['wilddet_match']=dict(score=float(wild_item['score']),score_2d=float(wild_item['score_2d']),
                score_3d=float(wild_item['score_3d']),bbox_xyxy=wild_item['bbox_xyxy'],
                box_camera_center_wlh_quaternion=wild_item['box_camera_center_wlh_quaternion'],
                association=wild_link)
        if yolo_item is not None:
            instance['yolo_detection_id']=int(yolo_item.get('id',-1));instance['yolo_match']=dict(
                score=float(yolo_item['score']),bbox_xyxy=yolo_item['bbox_xyxy'],association=yolo_link)
        if support['accepted'] and person_confirmed:
            confidence=_xy_confidence(float(base['score']),support['support_count'],support['xy_spread_m'])
            obj=dict(instance_id=int(iid),xy=support['center_xy'],confidence=confidence,xy_confidence=confidence,
                confidence_is_calibrated_probability=False,
                person_class_confirmation_sources=confirmation_sources,
                xy_source='SAM_person_instance_exclusive_LiDAR_coordinatewise_median',
                xy_method=support['position_method'],box_3d=None,box_3d_valid=False,
                dimensions_valid=False,yaw_valid=False,manual_ground_truth=False,
                supervision_mask=dict(x=True,y=True,z=False,length=False,width=False,height=False,yaw=False),
                sam_score=float(base['score']),wilddet_instance_id=instance['wilddet_instance_id'],
                yolo_detection_id=instance['yolo_detection_id'],wilddet_diagnostic_only=instance['wilddet_match'],
                yolo_comparison_only=instance['yolo_match'],points=int(support['support_count']),frames=[],
                support_spread_m=float(support['xy_spread_m']),support_bounds_xy=[support['xy_min_m'],support['xy_max_m']],
                support_source='all_exclusive_SAM_instance_mask_projected_visible_LiDAR_points',
                bbox_xyxy=base['bbox_xyxy'],reasons=[])
            accepted.append(obj);xy_records.append(obj)
            instance.update(status='accepted_xy_from_all_exclusive_mask_points',box_3d_valid=False,xy=obj['xy'],
                position_method=support['position_method'],points_used=int(support['support_count']))
        else:
            reasons=[]
            if not support['accepted']:reasons.append(support['reason'])
            if not person_confirmed:reasons.append('unconfirmed_by_yolo26x_person')
            rejected.append(dict(instance_id=int(iid),sam_score=float(base['score']),sam_bbox_xyxy=base['bbox_xyxy'],
                wilddet_instance_id=instance.get('wilddet_instance_id'),yolo_detection_id=instance.get('yolo_detection_id'),
                points=int(support.get('support_count',0)),reasons=reasons,manual_ground_truth=False,box_3d_valid=False))
            instance.update(status='ignore_unmatched_or_insufficient_support',reasons=reasons,box_3d_valid=False)
        support_idx=np.asarray(assigned,np.int64)
        support_indices.extend(support_idx.tolist());support_point_cloud.extend(xyz[support_idx].astype(np.float32).tolist())
        support_offsets.append(len(support_indices))
        instance['support_offset_start']=support_offsets[-2];instance['support_offset_end']=support_offsets[-1]
        instance_records.append(instance)
    # Raw YOLO/WildDet candidates without an assigned SAM mask remain explicit ignore records.
    matched_wild={x['wilddet_instance_id'] for x in instance_records if x['wilddet_instance_id'] is not None and x.get('wilddet_match')}
    unmatched_wild=[dict(wilddet_instance_id=int(d.get('wilddet_instance_id',i)),score=float(d['score']),score_3d=float(d['score_3d']),
        bbox_xyxy=d['bbox_xyxy'],reasons=['unmatched_sam_person_instance'],manual_ground_truth=False)
        for i,d in enumerate(eligible_wild) if int(d.get('wilddet_instance_id',i)) not in matched_wild]
    matched_yolo={x['yolo_detection_id'] for x in instance_records if x['yolo_detection_id'] is not None and x.get('yolo_match')}
    unmatched_yolo=[dict(yolo_detection_id=int(d.get('id',i)),score=float(d['score']),bbox_xyxy=d['bbox_xyxy'],
        reasons=['unmatched_sam_person_instance'],manual_ground_truth=False)
        for i,d in enumerate(persons) if int(d.get('id',i)) not in matched_yolo]
    # Preserve original negatives and separately provide effective negatives with every observed candidate-support cell ignored.
    neg_path=OLD_SUPERVISION/'frames'/f'{stem}.npz'
    with np.load(neg_path) as old:
        old_negative=old['negative_mask'].astype(bool).copy();old_xy=old['xy'].copy();old_conf=old['confidence'].copy()
    if old_negative.shape!=(80,80):raise ValueError('legacy negative-mask shape changed')
    ignore_mask=ignore_grid_from_points(ignored_ids,xyz,candidate_xy=candidate_xy)
    for item in xy_records:
        cell,ok=_grid_cells(np.asarray(item['xy'],float)[None,:])
        if ok[0]:ignore_mask[cell[0,1],cell[0,0]]=True
    effective_negative=old_negative&~ignore_mask
    return dict(frame=int(frame),frame_id=str(source_row['frame_id']),source_frame_id=int(source_row.get('source_frame_id',frame)),
        timestamp_ns=timestamp,instances=instance_records,accepted=accepted,rejected=rejected,
        wilddet_unmatched=unmatched_wild,yolo_unmatched=unmatched_yolo,
        wilddet_associations=wild_assoc,yolo_associations=yolo_assoc,
        legacy_negative_mask=old_negative,ignore_mask=ignore_mask,effective_negative_mask=effective_negative,
        old_xy=old_xy,old_confidence=old_conf,xy=xy_records,box_3d=box3d,box_3d_instance_ids=box3d_ids,
        box_3d_confidence=box3d_conf,box_3d_corners_lidar=np.asarray(box3d_corners,float).reshape(-1,8,3),support_indices=np.asarray(support_indices,np.int64),
        support_xyz=np.asarray(support_point_cloud,np.float32).reshape(-1,3),support_offsets=np.asarray(support_offsets,np.int64),
        point_owner=point_owner,overlap_cover_count=cover,ignored_point_indices=ignored_ids,
        legacy_label_point_ids=dict(person=int(np.count_nonzero(labels==1)),ground=int(np.count_nonzero(labels==0))),
        yolo_person_detections=persons,wilddet_person_detections=raw_wild)


def write_frame(result,output_root):
    out=Path(output_root);frame=int(result['frame']);stem=f'{frame:06d}'
    metadata={k:v for k,v in result.items() if k not in {'legacy_negative_mask','ignore_mask','effective_negative_mask',
        'old_xy','old_confidence','xy','box_3d','box_3d_instance_ids','box_3d_confidence','box_3d_corners_lidar','support_indices','support_xyz',
        'support_offsets','point_owner','overlap_cover_count','ignored_point_indices'}}
    metadata.update(schema='pedestrian_instance_labels_v3',manual_ground_truth=False,added_to_training=False,
        source_frame_id=str(result['frame_id']),source_frame_number=int(frame),timestamp_ns=int(result['timestamp_ns']),
        legacy_xy=result['old_xy'].astype(float).tolist(),legacy_xy_confidence=result['old_confidence'].astype(float).tolist(),
        policy={'person_confirmation':'require one-to-one YOLO26x person match; SAM/WildDet alone cannot confirm',
                'position_xy':'coordinatewise median over all exclusive visible valid ROI LiDAR points in each SAM3 person mask; no clustering, detector bbox cropping, or detector-center substitution; class confirmation is recorded separately',
                'box_3d':'not emitted in final XY labels; raw WildDet3D Mono3D boxes are separate diagnostics only',
                'dimensions_yaw_supervision':False,'xy_supervision':True,'unmatched_is_negative':False})
    arrays=dict(frame=np.asarray(frame,np.int32),frame_id=np.asarray(result['frame_id']),source_frame_id=np.asarray(int(frame),np.int32),
        timestamp_ns=np.asarray(result['timestamp_ns'],np.int64),xy=np.asarray([x['xy'] for x in result['xy']],np.float32).reshape(-1,2),
        confidence=np.asarray([x['confidence'] for x in result['xy']],np.float32),
        xy_instance_ids=np.asarray([x['instance_id'] for x in result['xy']],np.int32),
        box_3d=np.asarray(result['box_3d'],np.float32).reshape(-1,7),
        box_3d_corners_lidar=np.asarray(result['box_3d_corners_lidar'],np.float32).reshape(-1,8,3),
        box_3d_instance_ids=np.asarray(result['box_3d_instance_ids'],np.int32),
        box_3d_confidence=np.asarray(result['box_3d_confidence'],np.float32),
        legacy_negative_mask=np.asarray(result['legacy_negative_mask'],bool),
        ignore_mask=np.asarray(result['ignore_mask'],bool),effective_negative_mask=np.asarray(result['effective_negative_mask'],bool),
        support_indices=np.asarray(result['support_indices'],np.int32),support_xyz=np.asarray(result['support_xyz'],np.float32),
        support_offsets=np.asarray(result['support_offsets'],np.int32))
    dest=out/'frames'/f'{stem}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(dest)
    atomic_json(out/'metadata'/f'{stem}.json',metadata)
    return dest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--smoke',action='store_true');group.add_argument('--all',action='store_true');group.add_argument('--frames',type=int,nargs='+')
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--sam-root',type=Path,default=OUT/'raw_sam_person')
    parser.add_argument('--wild-root',type=Path,default=OUT/'raw_wilddet_person');args=parser.parse_args()
    rows=json.loads((SOURCE/'index.json').read_text())['rows'];by_frame={int(Path(r['path']).stem):r for r in rows}
    frames=choose_smoke_frames()['frames'] if args.smoke else sorted(by_frame) if args.all else sorted(set(args.frames))
    if args.all and len(frames)!=971:raise ValueError(f'Expected 971 source frames, got {len(frames)}')
    for root in (args.sam_root,args.wild_root):
        report=json.loads((root/'run_report.json').read_text())
        if report.get('stage')!='complete':
            raise RuntimeError(f'Incomplete inference cache: {root}')
        missing=[f'{f:06d}' for f in frames if not (root/'frames'/f'{f:06d}.npz' if root==args.sam_root else root/'frames'/f'{f:06d}.json').is_file()]
        if missing:raise RuntimeError(f'Inference cache {root} missing requested frames: {missing}')
    out=args.output.resolve();(out/'frames').mkdir(parents=True,exist_ok=True);(out/'metadata').mkdir(exist_ok=True);(out/'support').mkdir(exist_ok=True)
    wild_manifest=json.loads(WILDDET_WEIGHT_MANIFEST.read_text())
    policy=dict(schema='pedestrian_instance_labels_v3',sam_prompt=SAM3_PROMPT,sam_instance_score_min=SAM3_INSTANCE_SCORE_MIN,
        person_confirmation=dict(require_one_to_one_yolo26x_person_match=True,
            sam_or_wilddet_alone_can_confirm=False,position_uses_detector_box_or_center=False),
        sam_mask_pixel_threshold=SAM3_MASK_THRESHOLD,sam_checkpoint_sha256=sha256(SAM3_WEIGHTS),
        wilddet_prompt=WILDDET_PROMPT,wilddet_checkpoint_sha256=wild_manifest['sha256'],
        wilddet_combined_score_min=WILDDET_COMBINED_SCORE_MIN,wilddet_score_3d_min=WILDDET_SCORE_3D_MIN,
        yolo_cache=str(YOLO),yolo_person_score_min=YOLO_PERSON_SCORE_MIN,yolo_cache_policy_sha256=sha256(YOLO/'policy.json'),
        association=dict(box_iou_min=.1,mask_pixel_coverage_min=.25,ambiguity_utility_margin=.05,one_to_one=True),
        position=dict(method='coordinatewise median of all exclusive valid visible ROI LiDAR points per SAM instance',
                      minimum_support_points=MIN_SUPPORT_POINTS,clustering=False,uses_all_instance_points_without_detector_bbox_cropping=True,
                      detector_center_substitution=False,class_confirmation_is_separate=True,
                      confidence='SAM score times support reliability min(1,n/10)*exp(-median_xy_spread/0.4); uncalibrated'),
        support=dict(exclusive_sam_mask_points=True,duplicate_mask_pixels_ignored=True,minimum_points=MIN_SUPPORT_POINTS,
                     saved_point_indices='all instance-exclusive support points used for np.median(xyz[indices,:2])'),
        diagnostics='WildDet3D person Mono3D raw boxes retained separately; no depth-refined boxes in position labels',
        training_roi=list(TRAINING_ROI),negative_policy='preserve legacy negative mask; remove cells with candidate/unknown projected support and each new positive XY cell; no unmatched candidates treated as negatives',
        supervision_mask=dict(x=True,y=True,z=False,length=False,width=False,height=False,yaw=False),
        manual_ground_truth=False,added_to_training=False,source_index_sha256=sha256(SOURCE/'index.json'))
    atomic_json(out/'policy.json',policy)
    report_path=out/'status.json';atomic_json(report_path,dict(stage='running',pid=os.getpid(),frame_count=len(frames),frames=[f'{f:06d}' for f in frames],started=time.time()))
    xy_total=box_total=0;completed=[]
    try:
        for pos,frame in enumerate(frames,1):
            result=process_frame(frame,by_frame[frame],args.sam_root,args.wild_root,out)
            npz=write_frame(result,out);xy_total+=len(result['xy']);box_total+=len(result['box_3d']);completed.append(f'{frame:06d}')
            atomic_json(report_path,dict(stage='running',pid=os.getpid(),completed_frames=pos,frame_count=len(frames),
                xy_instances=xy_total,valid_3d_boxes=box_total,started=time.time(),updated=time.time()))
            if pos%20==0 or pos==len(frames):print(f'processed {pos}/{len(frames)} xy={xy_total} 3d={box_total}',flush=True)
        source_ids={Path(r['path']).stem:r['frame_id'] for r in rows}
        records=[]
        for frame in frames:
            stem=f'{frame:06d}';p=out/'frames'/f'{stem}.npz';meta=out/'metadata'/f'{stem}.json'
            with np.load(p) as z:
                nxy=len(z['xy']);n3=len(z['box_3d']);
            records.append(dict(frame=stem,frame_id=source_ids[stem],npz=f'frames/{stem}.npz',metadata=f'metadata/{stem}.json',
                npz_sha256=sha256(p),metadata_sha256=sha256(meta),xy_instances=nxy,box_3d_instances=n3))
        atomic_json(out/'index.json',dict(schema='pedestrian_instance_labels_v3',source_index_sha256=sha256(SOURCE/'index.json'),
            frames=records,frame_count=len(records),manual_ground_truth=False,added_to_training=False))
        summary=dict(stage='complete',pid=os.getpid(),completed_frames=len(completed),frame_count=len(frames),
            xy_instances=xy_total,valid_3d_boxes=box_total,manual_ground_truth=False,added_to_training=False,
            source_index_sha256=sha256(SOURCE/'index.json'),finished=time.time())
        atomic_json(report_path,summary);print(json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_json(report_path,dict(stage='failed',pid=os.getpid(),completed_frames=completed,error=repr(exc)));raise


if __name__=='__main__':main()
