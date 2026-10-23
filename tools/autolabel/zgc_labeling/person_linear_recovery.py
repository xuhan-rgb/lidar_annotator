"""Local linear consistency and current-frame mask-point recovery; review-only outputs."""
import numpy as np
from scipy.optimize import least_squares


def local_prediction(frames,times,xy,index,excluded,frame_radius=5,time_radius=.50001):
    current=frames[index]
    left=np.searchsorted(frames,current-frame_radius);right=np.searchsorted(frames,current+frame_radius,side='right')
    candidates=np.arange(left,right);dt=times[candidates]-times[index]
    use=(np.abs(dt)<=time_radius)&(candidates!=index)&~excluded[candidates]&(np.abs(dt)>.000001)
    neighbours=candidates[use];offsets=dt[use]
    if len(neighbours)<3:return None
    p=xy[neighbours];initial=np.r_[np.median(p,axis=0),[0.,0.]]
    def residual(x):return ((x[:2]+offsets[:,None]*x[2:]-p)/.2).ravel()
    fit=least_squares(residual,initial,bounds=([-np.inf,-np.inf,-3.,-3.],[np.inf,np.inf,3.,3.]),loss='soft_l1',max_nfev=60)
    if not fit.success:return None
    speed=np.linalg.norm(fit.x[2:])
    if speed>3.:
        fit.x[2:]*=3./speed
        velocity=fit.x[2:].copy()
        intercept=least_squares(lambda x:((x+offsets[:,None]*velocity-p)/.2).ravel(),fit.x[:2],loss='soft_l1',max_nfev=60)
        if not intercept.success:return None
        fit.x[:2]=intercept.x
    errors=np.linalg.norm(fit.x[:2]+offsets[:,None]*fit.x[2:]-p,axis=1)
    if (errors<=.8).sum()<max(3,int(np.ceil(.7*len(errors)))):return None
    return dict(xy=fit.x[:2],velocity=fit.x[2:],neighbour_indices=neighbours,neighbour_errors=errors,two_sided=bool(np.any(offsets<0) and np.any(offsets>0)))


def find_outliers(frames,times,xy,visual_presence_frames=None):
    frames=np.asarray(frames);times=np.asarray(times);xy=np.asarray(xy);excluded=np.zeros(len(frames),bool);predictions={}
    def predict(i):
        q=local_prediction(frames,times,xy,i,excluded)
        if q is None and visual_presence_frames is not None:
            q=local_prediction(frames,times,xy,i,excluded,frame_radius=10,time_radius=1.00001)
            if q is not None:
                neighbours=frames[q['neighbour_indices']];lo=min(int(frames[i]),int(neighbours.min()));hi=max(int(frames[i]),int(neighbours.max()))
                if not all(f in visual_presence_frames for f in range(lo,hi+1)):q=None
                else:q['extended_visual_continuity']=True
        return q
    for _ in range(2):
        new=[]
        for i in range(len(frames)):
            if excluded[i]:continue
            prediction=predict(i)
            if prediction is not None and np.linalg.norm(xy[i]-prediction['xy'])>.8:new.append(i);predictions[i]=prediction
        if not new:break
        excluded[new]=True
    for i in np.flatnonzero(excluded):
        refined=predict(i)
        if refined is not None:predictions[i]=refined
    return excluded,predictions


def recover_from_mask(points_world,source_ids,predicted_xy,radius=.8):
    points=np.asarray(points_world);ids=np.asarray(source_ids,int);keep=np.isfinite(points).all(1)&(np.linalg.norm(points[:,:2]-predicted_xy,axis=1)<=radius)
    if keep.sum()>=3:return np.median(points[keep],axis=0),ids[keep],'linear_guided_mask_lidar'
    return None,ids[keep],'linear_interpolation_only'


def camera_ray_recovery(original_point_lidar,expected_xy_map,points_lidar,source_ids,T,E,radius=.8):
    """Only camera-depth changes; preserve original full3D centre pixel exactly."""
    original=np.asarray(original_point_lidar,float);points=np.asarray(points_lidar,float);ids=np.asarray(source_ids,int)
    cam=E[:3,:3]@original+E[:3,3]
    if cam[2]<=.1:return None,np.empty(0,int),'ignored_invalid_camera_ray',dict(reason='original_depth_not_positive')
    inverse=np.linalg.inv(E);origin=inverse[:3,3];step=inverse[:3,:3]@(cam/cam[2]);world_origin=T[:3,:3]@origin+T[:3,3];world_step=T[:3,:3]@step
    denom=np.dot(world_step[:2],world_step[:2])
    if denom<1e-10:return None,np.empty(0,int),'ignored_invalid_camera_ray',dict(reason='ray_has_no_xy_direction')
    expected=np.asarray(expected_xy_map);d=float(np.dot(expected-world_origin[:2],world_step[:2])/denom);ray_expected=world_origin+d*world_step;mismatch=float(np.linalg.norm(ray_expected[:2]-expected))
    diag=dict(original_camera_depth_m=float(cam[2]),camera_ray_origin_lidar=origin.tolist(),camera_ray_step_lidar=step.tolist(),linear_camera_depth_m=d,linear_ray_mismatch_m=mismatch,linear_camera_ray_xy_map=ray_expected[:2].tolist())
    if d<=.1 or mismatch>radius:return None,np.empty(0,int),'ignored_camera_ray_conflict',diag
    world=points@T[:3,:3].T+T[:3,3];keep=np.isfinite(points).all(1)&(np.linalg.norm(world[:,:2]-ray_expected[:2],axis=1)<=radius)
    selected=ids[keep];diag['supported_camera_depths_m']=[]
    if keep.sum()>=3:
        depths=points[keep]@E[:3,:3].T+E[:3,3];depth=float(np.median(depths[:,2]));diag['supported_camera_depths_m']=depths[:,2].tolist();status='linear_guided_mask_lidar'
    else:depth=d;status='linear_interpolation_only'
    if depth<=.1:return None,selected,'ignored_camera_ray_conflict',diag
    point=origin+depth*step;diag['corrected_camera_depth_m']=depth
    return point,selected,status,diag
