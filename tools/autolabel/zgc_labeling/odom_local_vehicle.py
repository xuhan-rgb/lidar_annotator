"""Use recorded odom locally; keep geometrically valid single-frame boxes when propagation fails."""
from collections import Counter
import numpy as np
from anchor_vehicle_refine import select_anchor,refine_from_anchor
from temporal_refine import box_iou
from review_bev_pose import visible_edge_distances,projected_bbox


def keep_single_frame(o,reason):
    edges=visible_edge_distances(o['points'][:,:2],o['box'][:2],o['box'][6],o['box'][3:6])
    median=float(np.median(edges)) if len(edges) else float('inf')
    projected=projected_bbox(o['box'],o['K'],o['E'],(1920,1080))
    overlap=box_iou(projected,o['bbox']) if projected is not None else 0.
    usable=len(o['points'])>=10 and median<=.3 and overlap>=.3
    o['temporal_confirmed']=False
    o['single_frame_diagnostics']=dict(reason=reason,points=len(o['points']),edge_median_m=median,projection_iou=overlap)
    if usable:o['status']='accepted_single_frame';o['refined_box']=o['box'].copy()
    else:o['status']='ignore_single_frame_geometry';o.pop('refined_box',None)
    return usable


def refine_local_odom_tracks(tracks,params):
    statuses=Counter();max_age=params.get('local_anchor_max_age_s',2.)
    for track in tracks:
        observations=track['observations'];used=set()
        for o in observations:
            if len(observations)<params['min_track_observations']:
                keep_single_frame(o,'short_track');statuses[o['status']]+=1;continue
            nearby=[q for q in observations if abs(q['time']-o['time'])<=max_age+1e-6]
            selected=select_anchor(nearby)
            if selected is None:
                keep_single_frame(o,'no_reliable_local_anchor');statuses[o['status']]+=1;continue
            anchor,quality=selected;age=abs(anchor['time']-o['time']);box,diagnostics=refine_from_anchor(o,anchor)
            o['anchor_diagnostics']={**diagnostics,'anchor_age_s':age};o['anchor_selection']=quality
            if box is None:keep_single_frame(o,diagnostics['reason'])
            else:
                o['status']='accepted_temporal';o['temporal_confirmed']=True;o['refined_box']=box;used.add(anchor['frame'])
            statuses[o['status']]+=1
        track['local_anchor_frames']=sorted(used)
    return statuses
