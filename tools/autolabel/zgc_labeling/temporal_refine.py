"""Segment-local tracking: pedestrian foreground/outlier filtering, static vehicle ray/yaw fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'wilddet3d_zgc'))
from review_bev_pose import visible_edge_distances,wrap_angle
from label_geometry import boxes_to_corners_lidar

PARAMETERS=dict(depth_gap_m=.8,min_person_points=5,min_track_observations=3,max_gap_s=.5,
                pedestrian_max_speed_m_s=3.,pedestrian_position_slack_m=.8,vehicle_association_gate_m=2.5,
                optimization_window=9,static_center_scale_m=.3,static_yaw_scale_rad=.15,
                point_edge_scale_m=.15,maximum_vehicle_yaw_change_deg=90.,
                anchor_min_points=60,anchor_min_points_per_face=10,anchor_face_distance_m=.3,anchor_min_face_span_m=.5)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_iou(a,b):
    a,b=np.asarray(a),np.asarray(b);inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-6))


def foreground_indices(depth,gap=.8,minimum=5):
    """Nearest depth layer; never fall back to a farther wall when foreground is sparse."""
    order=np.argsort(depth);cuts=np.flatnonzero(np.diff(np.asarray(depth)[order])>gap)
    nearest=order[:int(cuts[0]+1)] if len(cuts) else order
    return nearest if len(nearest)>=minimum else np.empty(0,dtype=int)


def world_point(point,T):return T[:3,:3]@point+T[:3,3]


def world_heading(yaw,T):
    axis=T[:3,:3]@np.array([np.cos(yaw),np.sin(yaw),0.])
    return np.arctan2(axis[1],axis[0])


def track_observations(frames,kind,params):
    tracks=[];active=[];last_segment=None;statuses=Counter()
    for frame in frames:
        if frame['segment']!=last_segment:active=[]
        last_segment=frame['segment'];now=frame['time']
        active=[t for t in active if 0<now-t['observations'][-1]['time']<=params['max_gap_s']]
        observations=frame[kind];cost=np.full((len(active),len(observations)),1e6)
        for i,track in enumerate(active):
            previous=track['observations'][-1];dt=now-previous['time']
            velocity=np.zeros(3)
            if kind=='persons' and len(track['observations'])>=2:
                older=track['observations'][-2];velocity=(previous['world']-older['world'])/(previous['time']-older['time'])
                speed=np.linalg.norm(velocity)
                if speed>params['pedestrian_max_speed_m_s']:velocity*=params['pedestrian_max_speed_m_s']/speed
            predicted=previous['world']+velocity*dt
            gate=params['vehicle_association_gate_m'] if kind=='vehicles' else params['pedestrian_position_slack_m']+params['pedestrian_max_speed_m_s']*dt
            for j,obs in enumerate(observations):
                distance=np.linalg.norm(obs['world']-predicted);iou=box_iou(previous['bbox'],obs['bbox'])
                # Image association + ego-motion compensated geometry must both agree.
                if distance<=gate and iou>=.05:cost[i,j]=distance/gate+1-iou
        assignments={}
        if cost.size:
            ii,jj=linear_sum_assignment(cost)
            assignments={int(j):active[int(i)] for i,j in zip(ii,jj) if cost[i,j]<1e6}
        for j,obs in enumerate(observations):
            if j in assignments:track=assignments[j]
            else:
                # A bbox continuation with incompatible 3D evidence is an outlier, not a new identity.
                if any(box_iou(t['observations'][-1]['bbox'],obs['bbox'])>=.3 for t in active):
                    obs['status']='ignore_temporal_position_jump';statuses[obs['status']]+=1;continue
                track=dict(track_id=len(tracks),kind=kind,segment=frame['segment'],observations=[])
                tracks.append(track);active.append(track)
            obs['track_id']=track['track_id'];track['observations'].append(obs)
    for track in tracks:
        for obs in track['observations']:
            obs['status']='accepted_temporal' if len(track['observations'])>=params['min_track_observations'] else 'ignore_short_track'
            statuses[obs['status']]+=1
    return tracks,statuses



def vehicle_support_quality(points,box,params):
    c,s=np.cos(box[6]),np.sin(box[6]);R=np.array([[c,-s],[s,c]])
    q=(points[:,:2]-box[:2])@R;origin=-box[:2]@R;half=box[3:5]/2
    distances=[];axes=[]
    for axis in [0,1]:
        if abs(origin[axis])<=half[axis]:continue
        normal=q[:,axis]-np.sign(origin[axis])*half[axis]
        tangent=np.maximum(np.abs(q[:,1-axis])-half[1-axis],0.)
        distances.append(np.hypot(normal,tangent));axes.append(axis)
    faces=[]
    if distances and len(points):
        distance=np.stack(distances,axis=1);ownership=np.argmin(distance,axis=1)
        for i,axis in enumerate(axes):
            chosen=(ownership==i)&(distance[:,i]<=params['anchor_face_distance_m'])
            count=int(chosen.sum());span=float(np.ptp(q[chosen,1-axis])) if count else 0.
            faces.append(dict(axis=axis,points=count,span_m=span,qualified=count>=params['anchor_min_points_per_face'] and span>=params['anchor_min_face_span_m']))
    valid_faces=sum(f['qualified'] for f in faces)
    return dict(point_count=len(points),faces=faces,qualified_faces=valid_faces,reliable_anchor=len(points)>=params['anchor_min_points'])


def fit_static_window(observations,params):
    """Joint static centre/heading factors, with per-frame depth + yaw on frozen optical rays."""
    n=len(observations);boxes=np.array([o['box'] for o in observations]);yaw0=boxes[:,6]
    headings=np.array([world_heading(yaw,o['T']) for yaw,o in zip(yaw0,observations)])
    headings=headings[0]+np.array([wrap_angle(v-headings[0]) for v in headings])
    depth=[];origins=[];steps=[];lo_yaw=[];hi_yaw=[]
    for o,yaw in zip(observations,yaw0):
        E=o['E'];inverse=np.linalg.inv(E);camera=E[:3,:3]@o['box'][:3]+E[:3,3]
        depth.append(camera[2]);origins.append(inverse[:3,3]);steps.append(inverse[:3,:3]@(camera/camera[2]))
        reference=yaw-wrap_angle(yaw-o['initial_yaw']);lo_yaw.append(reference-np.pi/2);hi_yaw.append(reference+np.pi/2)
    depth=np.array(depth);origins=np.array(origins);steps=np.array(steps)
    world0=np.array([o['world'] for o in observations]);reliable=[i for i,o in enumerate(observations) if o.get('support_quality',{}).get('reliable_anchor',False)]
    anchor=world0[reliable[len(reliable)//2],:2] if reliable else world0[len(observations)//2,:2]
    x0=np.r_[depth,yaw0,anchor,headings[len(observations)//2]]
    lower=np.r_[np.maximum(.2,depth*.5),lo_yaw,[-np.inf]*3];upper=np.r_[depth*1.5,hi_yaw,[np.inf]*3]
    def unpack(x):return origins+x[:n,None]*steps,x[n:2*n]
    def residual(x):
        centres,yaws=unpack(x);r=[]
        for i,o in enumerate(observations):
            world=world_point(centres[i],o['T'])
            quality=2. if o.get('support_quality',{}).get('reliable_anchor',False) else .5
            r.extend(quality*(world[:2]-x[-3:-1])/params['static_center_scale_m'])
            if quality==2.:r.extend((x[-3:-1]-o['world'][:2])/.15)
            r.append(wrap_angle(world_heading(yaws[i],o['T'])-x[-1])/params['static_yaw_scale_rad'])
            edges=visible_edge_distances(o['points'][:,:2],centres[i,:2],yaws[i],o['box'][3:6])
            # Normalize per observation so a dense frame cannot dominate the track.
            r.extend(quality*edges/(params['point_edge_scale_m']*np.sqrt(max(1,len(edges)))))
            r.append(.1*(x[i]-depth[i])/max(.5,depth[i]*.1));r.append(.1*wrap_angle(yaws[i]-yaw0[i]))
        return np.asarray(r)
    before=residual(x0);result=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',max_nfev=50)
    after=residual(result.x)
    robust=lambda r:float(np.sum(np.sqrt(1+r*r)-1))
    if not result.success or robust(after)>robust(before)+1e-6:return None
    centres,yaws=unpack(result.x);final=boxes.copy();final[:,:3]=centres;final[:,6]=[wrap_angle(v) for v in yaws]
    return final,dict(objective_before=robust(before),objective_after=robust(after),solver_success=True,reliable_anchor_count=len(reliable))


def refine_vehicles(tracks,params):
    counts=Counter();radius=params['optimization_window']//2
    for track in tracks:
        obs=track['observations']
        if len(obs)<params['min_track_observations']:continue
        # Non-overlapping output blocks share neighbouring observations for boundary context.
        for start in range(0,len(obs),params['optimization_window']):
            stop=min(len(obs),start+params['optimization_window']);a=max(0,start-radius);b=min(len(obs),stop+radius)
            fitted=fit_static_window(obs[a:b],params)
            for i in range(start,stop):
                o=obs[i]
                if fitted is None:o['status']='ignore_static_fit_failed';counts[o['status']]+=1;continue
                boxes,diag=fitted;box=boxes[i-a];world=world_point(box[:3],o['T'])
                consensus=np.array([world_point(p[:3],q['T'])[:2] for p,q in zip(boxes,obs[a:b])])
                # Reject a ray/pose conflict rather than translating off the original image ray.
                deviation=np.linalg.norm(world[:2]-np.median(consensus,axis=0))
                heading=world_heading(box[6],o['T']);angles=np.array([world_heading(p[6],q['T']) for p,q in zip(boxes,obs[a:b])])
                angular_error=abs(wrap_angle(heading-np.arctan2(np.sin(angles).sum(),np.cos(angles).sum())))
                if deviation>.8 or angular_error>np.deg2rad(25):o['status']='ignore_static_pose_conflict'
                elif not (-20<=box[0]<20 and -20<=box[1]<20 and -3<=box[2]<6):o['status']='ignore_outside_roi'
                else:
                    o['refined_box']=box;o['diagnostics']={**diag,'static_centre_error_m':float(deviation),'static_heading_error_deg':float(np.degrees(angular_error))}
                counts[o['status']]+=1
    return counts



def lidar_yaw_from_world_heading(heading,T):
    direction=np.linalg.solve(T[:2,:2],np.array([np.cos(heading),np.sin(heading)]))
    return float(np.arctan2(direction[1],direction[0]))


def fit_shared_heading(tracks,params):
    """One optimized world heading for a static track, with every original 90-degree bound."""
    counts=Counter()
    for track in tracks:
        observations=[o for o in track['observations'] if o['status']=='accepted_temporal' and 'refined_box' in o]
        if not observations:continue
        reference=world_heading(observations[0]['refined_box'][6],observations[0]['T'])
        intervals=[]
        for o in observations:
            center=reference+wrap_angle(world_heading(o['initial_yaw'],o['T'])-reference)
            # Exact mapping of the permitted local heading interval into the world plane.
            a=center+wrap_angle(world_heading(o['initial_yaw']-np.pi/2,o['T'])-center)
            b=center+wrap_angle(world_heading(o['initial_yaw']+np.pi/2,o['T'])-center)
            intervals.append((min(a,b),max(a,b)))
        lower=max(a for a,b in intervals);upper=min(b for a,b in intervals)
        if upper-lower<np.deg2rad(5):
            for o in observations:o['status']='ignore_incompatible_directed_heading'
            counts['ignore_incompatible_directed_heading']+=len(observations);continue
        def residual(x):
            r=[]
            for o in observations:
                box=o['refined_box'];yaw=lidar_yaw_from_world_heading(x[0],o['T'])
                edges=visible_edge_distances(o['points'][:,:2],box[:2],yaw,box[3:6])
                quality=2. if o.get('support_quality',{}).get('reliable_anchor',False) else .5
                r.extend(quality*edges/(params['point_edge_scale_m']*np.sqrt(max(1,len(edges)))))
                r.append(.1*wrap_angle(world_heading(box[6],o['T'])-x[0]))
            return np.array(r)
        result=least_squares(residual,[np.clip(reference,lower+1e-8,upper-1e-8)],bounds=([lower],[upper]),loss='soft_l1',max_nfev=50)
        if not result.success:
            for o in observations:o['status']='ignore_shared_heading_fit_failed'
            continue
        for o in observations:
            yaw=lidar_yaw_from_world_heading(result.x[0],o['T'])
            if abs(wrap_angle(yaw-o['initial_yaw']))>np.pi/2+1e-8:
                o['status']='ignore_incompatible_directed_heading';continue
            o['refined_box'][6]=yaw
            o['diagnostics']['shared_world_heading_rad']=float(wrap_angle(result.x[0]))
            counts['shared_heading_accepted']+=1
    return counts


def load_frames(config,selected,params):
    source=Path(config['source_index']);rows=json.loads(source.read_text())['rows']
    with np.load(config['pose_npz']) as z:transforms=z['transforms'];segments=z['segments']
    with np.load(config['odometry_npz']) as z:timestamps=z['timestamps']
    pose_report=json.loads(Path(config['pose_report']).read_text());assert len(transforms)==len(rows)==len(timestamps)==len(pose_report['diagnostics'])
    np.testing.assert_allclose(transforms[:,:3,:3].transpose(0,2,1)@transforms[:,:3,:3],np.broadcast_to(np.eye(3),(len(rows),3,3)),atol=1e-5)
    person=Path(config['person_root']);vehicle=Path(config['vehicle_root']);frames=[];statistics=Counter()
    for index,row in enumerate(rows):
        frame=int(Path(row['path']).stem)
        if selected is not None and frame not in selected:continue
        with np.load(source.parent/row['path']) as z:
            xyz=z['xyz'].astype(float);depth=z['camera_depth'].astype(float);E=z['cam_extrinsic'].astype(float);K=z['cam_intrinsic'].astype(float);timestamp=int(z['lidar_timestamp_ns'])
        assert timestamp==int(timestamps[index]);T=transforms[index]
        entry=dict(frame=frame,time=timestamp/1e9,timestamp_ns=timestamp,segment=int(segments[index]),persons=[],vehicles=[],source=row,source_xyz=xyz)
        pm=json.loads((person/'metadata'/f'{frame:06d}.json').read_text())
        with np.load(person/'frames'/f'{frame:06d}.npz') as z:
            support=z['support_indices'];offset=z['support_offsets']
            for item in pm['accepted']:
                iid=int(item['instance_id']);ids=support[offset[iid]:offset[iid+1]]
                nearest=foreground_indices(depth[ids],params['depth_gap_m'],params['min_person_points'])
                record=dict(kind='person',frame=frame,time=entry['time'],instance_id=iid,bbox=pm['instances'][iid]['sam_bbox_xyxy'],baseline_xy=item['xy'],support_ids=ids[nearest],input_count=len(ids),status='pending')
                if len(nearest)<params['min_person_points']:record['status']='ignore_insufficient_foreground'
                else:
                    point=np.median(xyz[ids[nearest]],axis=0);record.update(point=point,world=world_point(point,T),xy=point[:2])
                    if len(nearest)<len(ids):statistics['person_foreground_selection_changed']+=1
                entry['persons'].append(record)
        vm=json.loads((vehicle/'refined'/f'{frame:06d}.json').read_text());records={int(d['prediction_index']):d for d in vm['all_detections']}
        with np.load(vehicle/'refined'/f'{frame:06d}.npz') as z:
            boxes=z['box_3d'].astype(float);seeds=z['seed_ids'];initial=z['initial_directed_yaw'];ids=z['support_point_indices'];offset=z['support_offsets']
            for i,box in enumerate(boxes):
                seed=int(seeds[i]);points=xyz[ids[offset[i]:offset[i+1]]]
                entry['vehicles'].append(dict(kind='vehicle',frame=frame,time=entry['time'],seed_id=seed,bbox=records[seed]['bbox_xyxy'],box=box,support_quality=vehicle_support_quality(points,box,params),initial_yaw=float(initial[i]),world=world_point(box[:3],T),T=T,E=E,K=K,points=points,support_ids=ids[offset[i]:offset[i+1]],status='pending'))
        frames.append(entry)
    return frames,statistics


def serializable(record):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in record.items() if k not in ['points','T','E','K']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--frames',nargs='+',type=int);args=parser.parse_args()
    config=json.loads(args.config.read_text());params={**PARAMETERS,**config.get('temporal_parameters',{})};output=Path(config['temporal_output'])
    if args.frames is not None:output=output/'review_subset'
    for name in ['frames','metadata']:(output/name).mkdir(parents=True,exist_ok=True)
    started=time.time();frames,stats=load_frames(config,set(args.frames) if args.frames else None,params)
    skipped=[]
    for frame in frames:
        invalid=[o for o in frame['persons'] if 'world' not in o];skipped.extend(invalid);frame['persons']=[o for o in frame['persons'] if 'world' in o]
    persons,pstatus=track_observations(frames,'persons',params);vehicles,vstatus=track_observations(frames,'vehicles',params)
    strategy=config.get('vehicle_strategy','joint_shared_heading')
    if strategy=='best_anchor':
        from anchor_vehicle_refine import refine_tracks_from_best_anchor
        refined_status=refine_tracks_from_best_anchor(vehicles,params);heading_status={}
    elif strategy=='local_odom_anchor':
        from odom_local_vehicle import refine_local_odom_tracks,keep_single_frame
        refined_status=refine_local_odom_tracks(vehicles,params);heading_status={}
        for f in frames:
            for o in f['vehicles']:
                if o['status']=='ignore_temporal_position_jump':keep_single_frame(o,'temporal_association_conflict')
    elif strategy=='joint_shared_heading':
        refined_status=refine_vehicles(vehicles,params);heading_status=fit_shared_heading(vehicles,params)
    else:raise ValueError(f'Unknown vehicle strategy: {strategy}')
    accepted_persons=accepted_vehicles=0
    for frame in frames:
        ps=[o for o in frame['persons'] if o['status']=='accepted_temporal'];vs=[o for o in frame['vehicles'] if o['status'] in ['accepted_temporal','accepted_single_frame'] and 'refined_box' in o]
        accepted_persons+=len(ps);accepted_vehicles+=len(vs);boxes=np.array([o['refined_box'] for o in vs],np.float32).reshape(-1,7)
        offsets=[0];support=[]
        for o in ps+vs:support.extend(o['support_ids'].tolist());offsets.append(len(support))
        np.savez_compressed(output/'frames'/f'{frame["frame"]:06d}.npz',frame=frame['frame'],timestamp_ns=frame['timestamp_ns'],person_xy=np.array([o['xy'] for o in ps],np.float32).reshape(-1,2),person_instance_ids=[o['instance_id'] for o in ps],person_track_ids=[o['track_id'] for o in ps],vehicle_box_3d=boxes,vehicle_corners_3d_lidar=boxes_to_corners_lidar(boxes).astype(np.float32),vehicle_seed_ids=[o['seed_id'] for o in vs],vehicle_track_ids=[o.get('track_id',-1) for o in vs],vehicle_temporal_confirmed=np.array([o.get('temporal_confirmed',o['status']=='accepted_temporal') for o in vs],bool),support_point_indices=support,support_offsets=offsets,manual_gt=False,added_to_training=False)
        observations=frame['persons']+frame['vehicles']+[o for o in skipped if o['frame']==frame['frame']]
        meta=dict(frame=frame['frame'],segment=frame['segment'],observations=[serializable(o) for o in observations],labels_source='temporal output NPZ',manual_gt=False,added_to_training=False)
        (output/'metadata'/f'{frame["frame"]:06d}.json').write_text(json.dumps(meta,indent=2))
    tracks=[dict(track_id=t['track_id'],kind=t['kind'],segment=t['segment'],anchor_frame=t.get('anchor_frame'),anchor_selection=t.get('anchor_selection'),local_anchor_frames=t.get('local_anchor_frames',[]),reliable_anchor_frames=[o['frame'] for o in t['observations'] if o.get('support_quality',{}).get('reliable_anchor',False)],observations=[dict(frame=o['frame'],status=o['status']) for o in t['observations']]) for t in persons+vehicles]
    (output/'tracks.json').write_text(json.dumps(tracks,indent=2))
    all_person_status=Counter(o['status'] for f in frames for o in f['persons']);all_person_status.update(o['status'] for o in skipped)
    all_vehicle_status=Counter(o['status'] for f in frames for o in f['vehicles'])
    report=dict(status='complete',vehicle_strategy=strategy,frames=len(frames),person_xy=accepted_persons,vehicle_boxes=accepted_vehicles,person_tracks=len(persons),vehicle_tracks=len(vehicles),person_status=dict(all_person_status),vehicle_refinement_status=dict(all_vehicle_status),shared_heading_status=dict(heading_status),statistics=dict(stats),parameters=params,elapsed_seconds=round(time.time()-started,3),source_index_sha256=sha(config['source_index']),pose_sha256=sha(config['pose_npz']),configuration=config,baseline_labels_modified=False,manual_gt=False,added_to_training=False,limits='Experimental temporal filtering; no manual GT; pose source and frame assumptions recorded in configuration; no cross-segment propagation or missing-detection label synthesis')
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='configuration'},indent=2))


if __name__=='__main__':main()
