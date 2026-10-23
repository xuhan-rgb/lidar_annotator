"""Select one independently good vehicle observation, propagate it, then make bounded corrections."""
import numpy as np
from scipy.optimize import least_squares
from temporal_refine import world_point,world_heading,lidar_yaw_from_world_heading,box_iou,wrap_angle
from review_bev_pose import visible_edge_distances,projected_bbox


def anchor_score(o):
    q=o['support_quality']
    if not q['reliable_anchor']:return None
    distances=visible_edge_distances(o['points'][:,:2],o['box'][:2],o['box'][6],o['box'][3:6])
    error=float(np.median(distances))
    projected=projected_bbox(o['box'],o['K'],o['E'],(1920,1080))
    overlap=box_iou(projected,o['bbox']) if projected is not None else 0.
    truncated=o['bbox'][0]<.03*1920 or o['bbox'][2]>.97*1920 or o['bbox'][1]<.03*1080 or o['bbox'][3]>.97*1080
    score=np.log1p(q['point_count'])*overlap/(1+error/.15)*(0.5 if truncated else 1.)
    return dict(score=float(score),edge_median_m=error,projection_iou=overlap,truncated=bool(truncated),points=q['point_count'],qualified_faces=q['qualified_faces'])


def select_anchor(observations):
    candidates=[(o,anchor_score(o)) for o in observations]
    candidates=[(o,q) for o,q in candidates if q is not None]
    if not candidates:return None
    # Ranking picks one observation. Other frames cannot change its pose by averaging.
    return max(candidates,key=lambda item:(item[1]['score'],-item[0]['frame']))


def refine_from_anchor(o,anchor,depth_adjustment_m=.25,yaw_adjustment_deg=5.):
    if o is anchor:return anchor['box'].copy(),dict(anchor_frame=anchor['frame'],anchor_pose_unchanged=True)
    inv=np.linalg.inv(o['T']);anchor_world=world_point(anchor['box'][:3],anchor['T'])
    expected=inv[:3,:3]@anchor_world+inv[:3,3]
    heading=world_heading(anchor['box'][6],anchor['T']);yaw=lidar_yaw_from_world_heading(heading,o['T'])
    # Keep the previously requested pixel-centre constraint; disclose incompatible rays.
    E=o['E'];inverse=np.linalg.inv(E);camera=E[:3,:3]@o['box'][:3]+E[:3,3]
    origin=inverse[:3,3];step=inverse[:3,:3]@(camera/camera[2])
    depth=float(np.dot(expected-origin,step)/np.dot(step,step))
    mismatch=float(np.linalg.norm(origin+depth*step-expected))
    if depth<=.2 or mismatch>.8:return None,dict(reason='anchor_incompatible_with_original_pixel_ray',ray_mismatch_m=mismatch)
    original=yaw-wrap_angle(yaw-o['initial_yaw']);limit=np.deg2rad(yaw_adjustment_deg)
    lower=max(yaw-limit,original-np.pi/2);upper=min(yaw+limit,original+np.pi/2)
    if lower>=upper:return None,dict(reason='anchor_incompatible_with_directed_yaw_bound')
    def decode(x):
        box=o['box'].copy();box[:3]=origin+x[0]*step;box[6]=wrap_angle(x[1]);return box
    def residual(x):
        box=decode(x);edges=visible_edge_distances(o['points'][:,:2],box[:2],box[6],box[3:6])
        r=list(edges/(.15*np.sqrt(max(1,len(edges)))))
        r.extend((box[:3]-expected)/.2);r.append(wrap_angle(box[6]-yaw)/np.deg2rad(3))
        projected=projected_bbox(box,o['K'],E,(1920,1080))
        if projected is not None:
            scale=np.tile(np.maximum(np.asarray(o['bbox'])[2:]-np.asarray(o['bbox'])[:2],40.)*.1,2)
            r.extend(.25*(projected-o['bbox'])/scale)
        return np.array(r)
    initial=np.array([depth,np.clip(yaw,lower+1e-8,upper-1e-8)])
    fit=least_squares(residual,initial,bounds=([max(.2,depth-depth_adjustment_m),lower],[depth+depth_adjustment_m,upper]),loss='soft_l1',max_nfev=60)
    if not fit.success:return None,dict(reason='anchor_local_fit_failed')
    box=decode(fit.x)
    if not np.all((box[:3]>=[-20,-20,-3])&(box[:3]<[20,20,6])):return None,dict(reason='outside_roi')
    error=np.median(visible_edge_distances(o['points'][:,:2],box[:2],box[6],box[3:6]))
    return box,dict(anchor_frame=anchor['frame'],ray_mismatch_m=mismatch,camera_depth_adjustment_m=float(fit.x[0]-depth),yaw_adjustment_from_anchor_deg=float(np.degrees(wrap_angle(box[6]-yaw))),edge_median_m=float(error))


def refine_tracks_from_best_anchor(tracks,params):
    from collections import Counter
    statuses=Counter()
    for track in tracks:
        observations=track['observations']
        if len(observations)<params['min_track_observations']:
            statuses['ignore_short_track']+=len(observations);continue
        selected=select_anchor(observations)
        if selected is None:
            for o in observations:o['status']='ignore_no_reliable_anchor';o.pop('refined_box',None)
            statuses['ignore_no_reliable_anchor']+=len(observations);continue
        anchor,quality=selected;track['anchor_frame']=anchor['frame'];track['anchor_selection']=quality
        for o in observations:
            box,diagnostics=refine_from_anchor(o,anchor)
            o['anchor_diagnostics']=diagnostics;o['anchor_selection']=quality
            if box is None:
                o['status']='ignore_anchor_pose_conflict';o.pop('refined_box',None)
            else:o['status']='accepted_temporal';o['refined_box']=box
            statuses[o['status']]+=1
    return statuses
