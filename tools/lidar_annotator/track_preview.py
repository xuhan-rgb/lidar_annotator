"""Read-only instance support filtered by optimized pedestrian positions."""
import json
import numpy as np
from .display_range import DEFAULT_LIMITS,point_mask
from .person_support import select_person_support
from tools.autolabel.zgc_labeling.finalize_person_positions import load_positions


def vehicle_boxes_in_frame(boxes, pose):
    """Transform saved map boxes into the current LiDAR frame."""
    boxes=np.asarray(boxes,float).reshape(-1,7).copy();inverse=np.linalg.inv(pose)
    boxes[:,:3]=boxes[:,:3]@inverse[:3,:3].T+inverse[:3,3]
    boxes[:,6]+=np.arctan2(inverse[1,0],inverse[0,0])
    return boxes


def vehicle_support_mask(xyz, boxes):
    """Use the same 3D support margin as the saved optimized review renderer."""
    support=np.zeros(len(xyz),bool)
    for box in boxes:
        delta=xyz-box[:3];co,si=np.cos(box[6]),np.sin(box[6])
        local=np.c_[co*delta[:,0]+si*delta[:,1],-si*delta[:,0]+co*delta[:,1],delta[:,2]]
        support|=np.isfinite(local).all(1)&(np.abs(local)<=box[3:6]/2+.12).all(1)
    return support


def project_points(xyz,extrinsic,intrinsic,shape):
    """LiDAR metres -> image pixels, excluding behind-camera/out-of-image points."""
    camera=xyz@extrinsic[:3,:3].T+extrinsic[:3,3]
    homogeneous=camera@intrinsic.T
    uv=np.full((len(xyz),2),np.nan)
    positive=np.isfinite(camera).all(1)&(camera[:,2]>.1)
    uv[positive]=homogeneous[positive,:2]/homogeneous[positive,2:3]
    h,w=shape
    valid=positive&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,valid


def load_targets(paths,arrays,records,limits):
    """Associate masks and saved centres by display ID, never by row position."""
    mask_path=paths['instance_masks']
    if mask_path.is_file():
        with np.load(mask_path,allow_pickle=False) as z:
            shape=tuple(z['mask_shape']);masks=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)
        if len(masks)!=len(records):raise ValueError('实例 mask 与检测记录数量不一致')
        objects=[dict(record,mask=mask) for record,mask in zip(records,masks)]
    elif paths['track_masks'].is_file():
        with np.load(paths['track_masks'],allow_pickle=False) as z:
            shape=tuple(z['mask_shape']);masks=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)
            objects=[dict(display_id=int(tid),prompt='person',mask=mask) for tid,mask in zip(z['object_ids'],masks)]
    else:return []
    uv=arrays['pixels_uv'];xyz=arrays['xyz'];h,w=shape
    pixels=np.floor(np.where(np.isfinite(uv),uv,0)).astype(int)
    visible=arrays.get('visible',arrays.get('in_image'))
    # Semantic decisions use the processing ROI, never the user's display bounds.
    roi=np.asarray(arrays.get('training_roi_mask',point_mask(xyz,DEFAULT_LIMITS)),bool)
    valid=roi&np.isfinite(xyz).all(1)&visible&np.isfinite(uv).all(1)&(arrays['camera_depth']>0)&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    indices=np.flatnonzero(valid);x,y=pixels[indices].T
    cover=masks[:,y,x].sum(0)
    display=point_mask(xyz,limits)
    labels=arrays.get('point_raw_labels',arrays.get('point_labels'))
    mask_file=paths.get('mask')
    if labels is None and mask_file is not None and mask_file.is_file():
        from PIL import Image
        with Image.open(mask_file) as image:semantic=np.asarray(image)
        labels=np.full(len(xyz),255,np.uint8);labels[indices]=semantic[y,x]
    centres={}
    if paths['tracks'].is_file():
        z=load_positions(paths['tracks'])
        if 'original_point_lidar' in z:
            for i in np.flatnonzero(z['frame_ids']==paths['local']):
                centres[int(z['sam_video_ids'][i])]=(z['original_point_lidar'][i],z['corrected_point_lidar'][i],bool(z['position_available'][i]))
    radius=.5
    report=paths['tracks'].parent/'report.json'
    if report.is_file():radius=float(json.loads(report.read_text()).get('parameters',{}).get('reference_tube_m',radius))
    saved={}
    for path in sorted((paths['tracks'].parent/'metadata').glob(f"{paths['local']:06d}_I*.json")):
        item=json.loads(path.read_text())
        if item.get('frame')==paths['local'] and 'sam_video_id' in item:
            tid=int(item['sam_video_id']);saved[tid]=item if tid not in saved else None
    person_instances=[]
    metadata_path=paths.get('person_metadata')
    if metadata_path is not None and metadata_path.is_file():
        person_instances=json.loads(metadata_path.read_text()).get('instances',[])
    display_to_native={}
    id_map_path=paths.get('id_map')
    if id_map_path is not None and id_map_path.is_file():
        native_to_display=json.loads(id_map_path.read_text())
        for native,shown_id in native_to_display.items():
            display_to_native.setdefault(int(shown_id),[]).append(int(native))

    def class_confirmation(record, display_id):
        native=record.get('source_instance_id')
        if native is None:native=record.get('track_id')
        if native is None:
            mapped=display_to_native.get(int(display_id))
            native=(mapped[0] if mapped is not None and len(mapped)==1 else
                    (int(display_id) if id_map_path is None or not id_map_path.is_file() else None))
        if native is None:return None
        matches=[item for item in person_instances
                 if item.get('sam_source_instance_id') is not None and
                 str(item['sam_source_instance_id'])==str(native)]
        if len(matches)!=1 or 'person_class_confirmed' not in matches[0]:return None
        item=matches[0]
        if not item['person_class_confirmed']:return False
        return bool((item.get('yolo_match') or {}).get('score',0)>=.4)

    targets=[]
    for record in objects:
        tid=record.get('display_id')
        if tid is None:continue
        mask=record['mask'];support=indices[mask[y,x]&(cover==1)]
        points=xyz[support].copy();original=np.median(points,axis=0) if len(points) else None;corrected=None
        if record.get('prompt')=='person' and int(tid) in centres:
            before,after,available=centres[int(tid)]
            if np.isfinite(before).all():original=before.copy()
            if available and np.isfinite(before).all() and np.isfinite(after).all():corrected=after.copy()
        selection=None
        class_confirmed=None
        if record.get('prompt')=='person':
            class_confirmed=class_confirmation(record,tid)
            evidence=saved.get(int(tid));selection_center=corrected
            if int(tid) in saved and (evidence is None or evidence.get('position_available') is False):selection_center=None
            selection=select_person_support(xyz,support,labels,selection_center,
                                            evidence.get('selected_support_ids') if evidence else None,radius)
            if class_confirmed is False and labels is not None:
                person_support=support[np.asarray(labels)[support]==1]
                selection.update(kept_indices=np.empty(0,dtype=np.int64),
                                 rejected_indices=np.asarray(person_support,dtype=np.int64),
                                 status='unconfirmed_person_instance',review_required=True,saved_support_used=False)
            kept=selection['kept_indices'];rejected=selection['rejected_indices']
        else:kept=np.empty(0,int);rejected=np.empty(0,int)
        shown=support[display[support]];kept_shown=kept[display[kept]]
        targets.append(dict(optimized_indices=kept_shown,optimized_points=xyz[kept_shown].copy(),rejected_indices=rejected,radius=radius,id=int(tid),prompt=record.get('prompt'),indices=shown,points=xyz[shown].copy(),mask=mask,
                            semantic_selection=selection,person_class_confirmed=class_confirmed,
                            original=original,corrected=corrected,bbox=record.get('bbox_xyxy'),
                            color=(0,255,255) if record.get('prompt')=='person' else (255,155,30)))
    return targets


def bounds(points):
    if not len(points):return None
    return np.array([points.min(0),points.max(0)])


def track_view(data,target_id=None,stage='before'):
    """Compare raw mask support and inliers near the saved optimized XY centre."""
    result=dict(data);layers=[];arrows=[];focus_camera=[];focus_bev=[]
    rgb=data['track_rgb'].copy()
    selected=[t for t in data['targets'] if target_id is None or t['id']==target_id]
    missing=0
    for target in selected:
        original=target['original'];corrected=target['corrected'];points=target['points']
        color=target['color'];mask=target['mask']
        rgb[mask]=(data['track_rgb'][mask]*.65+np.array(color)*.35).astype(np.uint8)
        versions=[('before',points,original,(155,155,155) if stage=='compare' else color)] if stage!='after' else []
        if corrected is not None and original is not None:
            retained=target['optimized_points']
            if stage!='before':versions.append(('after',retained,corrected,color))
            # A common view for all stages preserves the camera while toggling.
            for cloud in (points,retained):
                keep=point_mask(cloud,data['limits']);projected,visible=project_points(cloud,data['extrinsic'],data['intrinsic'],rgb.shape[:2])
                focus_bev.extend(cloud[keep,:2]);focus_camera.extend(projected[keep&visible])
            if stage=='compare':arrows.append((original,corrected))
        else:
            missing+=1
            focus_bev.extend(points[:,:2])
            uv,valid=project_points(points,data['extrinsic'],data['intrinsic'],rgb.shape[:2]);focus_camera.extend(uv[valid])
        for name,cloud,center,tint in versions:
            keep=point_mask(cloud,data['limits']);uv,visible=project_points(cloud,data['extrinsic'],data['intrinsic'],rgb.shape[:2])
            layers.append(dict(id=target['id'],version=name,color=tint,bev=cloud[keep,:2],camera=uv[keep&visible],center=center))
    from PySide6.QtGui import QImage
    result['image']=QImage(rgb.tobytes(),rgb.shape[1],rgb.shape[0],rgb.shape[1]*3,QImage.Format.Format_RGB888).copy()
    result.update(track_layers=layers,track_arrows=arrows,records=[] if data['targets'] else data['records'],centres=np.empty((0,2)),centre_ids=[],boxes=[],box_ids=[],camera_lines=[],camera_centres=[],
                  camera_colors=np.full_like(data['camera_colors'],65),bev_colors=np.full_like(data['bev_colors'],65),palette=np.array([[65,65,65]],np.uint8),
                  focus_camera=bounds(np.asarray(focus_camera)),focus_bev=bounds(np.asarray(focus_bev)))
    selected_indices=np.concatenate([t['indices'] for t in selected]) if selected else np.empty(0,int)
    for view in ('camera','bev'):
        background=~np.isin(data[view+'_point_indices'],selected_indices)
        result[view+'_points']=data[view+'_points'][background]
        result[view+'_colors']=result[view+'_colors'][background]
    names={'before':'优化前','after':'优化后（已校验支持点）','compare':'前灰 / 后类别色；已校验支持点'}
    note=f" · {names[stage]} · 目标 {len(selected)} 个 · mask 支持点 {sum(len(t['points']) for t in selected)}"
    if stage!='before':note+=f" · 保留 {sum(len(t['optimized_points']) for t in selected)} / 忽略 {sum(len(t['rejected_indices']) for t in selected)}"
    reviews=sum(bool((t.get('semantic_selection') or {}).get('review_required')) and t.get('person_class_confirmed') is not False for t in selected)
    if reviews and stage!='before':note+=f' · {reviews} 个点团仍待核查'
    unconfirmed=sum(t.get('person_class_confirmed') is False for t in selected)
    if unconfirmed and stage!='before':note+=f' · {unconfirmed} 个类别未确认候选，行人语义已标忽略'
    if not data['targets']:note+=' · 当前帧缺少带 ID 的实例 mask，无法提取目标点云'
    if missing and stage!='before':note+=f' · {missing} 个目标无可用优化结果，未生成优化后点集'
    if target_id is not None and not selected:note+=' · 当前帧无此 ID'
    result['status']=data['status']+note
    return result


def ignored_support_indices(targets):
    rejected=[target['rejected_indices'] for target in targets if len(target['rejected_indices'])]
    return np.concatenate(rejected) if rejected else np.empty(0,int)


def person_support_audit(targets):
    items=[]
    for target in targets:
        selection=target.get('semantic_selection')
        if target.get('prompt')!='person' or selection is None:continue
        items.append(dict(track_id=target['id'],status=selection['status'],review_required=selection['review_required'],
                          class_confirmed=target.get('person_class_confirmed'),
                          semantic_center_xy=selection['center_xy'],saved_support_used=selection['saved_support_used'],
                          radius_m=target.get('radius',.5),minimum_person_points=5,minimum_core_fraction=.5,maximum_center_shift_m=.75,
                          fallback_max_distance_m=selection.get('fallback_max_distance_m'),
                          kept_indices=selection['kept_indices'].tolist(),ignored_indices=selection['rejected_indices'].tolist(),
                          old_class=1,new_class=255))
    return dict(policy='validated_person_support_v3',only_person_to_ignore=True,xyz_modified=False,
                instances=items,ignored_indices=np.unique(ignored_support_indices(targets)).tolist())
