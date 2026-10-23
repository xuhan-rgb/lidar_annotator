"""Banded robust trajectory reference, with no pairwise point clustering."""
import numpy as np
from scipy.sparse import diags,lil_matrix
from scipy.sparse.linalg import spsolve


def robust_reference(times,xy,quality,iterations=30):
    times=np.asarray(times,float);xy=np.asarray(xy,float);quality=np.clip(np.asarray(quality,float),.01,1.);n=len(xy)
    if n<3:return xy.copy(),dict(iterations=0,converged=True)
    dt=np.diff(times)
    if np.any(dt<=0):raise ValueError('Reference observations must be time ordered')
    # Velocity variation on non-uniform timestamps; D has three entries per row.
    D=lil_matrix((n-2,n),dtype=float)
    for i in range(n-2):
        average=(dt[i]+dt[i+1])/2
        D[i,i]=1/(dt[i]*average);D[i,i+1]=-(1/dt[i]+1/dt[i+1])/average;D[i,i+2]=1/(dt[i+1]*average)
    D=D.tocsr();B=lil_matrix((n-1,n),dtype=float)
    raw_velocity=np.diff(xy,axis=0)/dt[:,None];velocity_prior=np.zeros_like(raw_velocity)
    for i in range(n-1):
        B[i,i]=-1/dt[i];B[i,i+1]=1/dt[i]
        local=raw_velocity[max(0,i-5):min(n-1,i+6)];local=local[np.linalg.norm(local,axis=1)<=3.]
        if len(local):velocity_prior[i]=np.median(local,axis=0)
    B=B.tocsr();penalty=D.T@D/(3.**2)+B.T@B/(.5**2);velocity_rhs=B.T@velocity_prior/(.5**2);reference=xy.copy();converged=False
    for step in range(iterations):
        residual=np.linalg.norm(reference-xy,axis=1)/.2
        weights=quality/(np.sqrt(1+residual**2)*.2**2)
        matrix=diags(weights)+penalty
        updated=np.c_[spsolve(matrix,weights*xy[:,0]+velocity_rhs[:,0]),spsolve(matrix,weights*xy[:,1]+velocity_rhs[:,1])]
        delta=float(np.max(np.linalg.norm(updated-reference,axis=1)));reference=updated
        if delta<1e-5:converged=True;break
    return reference,dict(iterations=step+1,converged=converged,max_update_m=delta)


def depth_on_reference_tube(original_point_lidar,reference_xy,T,E,points_lidar,source_ids,tube=.5):
    from person_linear_recovery import camera_ray_recovery
    point,ids,status,diag=camera_ray_recovery(original_point_lidar,reference_xy,points_lidar,source_ids,T,E,radius=tube)
    if point is None:return point,ids,status,diag
    origin=np.array(diag['camera_ray_origin_lidar']);step=np.array(diag['camera_ray_step_lidar']);world_step=T[:3,:3]@step
    maximum=np.sqrt(max(0,tube*tube-diag['linear_ray_mismatch_m']**2))/np.linalg.norm(world_step[:2]);depth=diag['corrected_camera_depth_m'];centre=diag['linear_camera_depth_m']
    bounded=float(np.clip(depth,max(.1,centre-maximum),centre+maximum));diag['unconstrained_supported_depth_m']=depth;diag['corrected_camera_depth_m']=bounded;diag['depth_clipped_to_reference']=abs(depth-bounded)>1e-8
    point=origin+bounded*step
    if status=='linear_guided_mask_lidar' and diag['depth_clipped_to_reference']:status='mask_supported_reference_constrained'
    return point,ids,status,diag


def static_reference(times,reference_xy,quality,origins,steps):
    """Use one stationary centre only when motion and diverse camera rays both support it."""
    times=np.asarray(times,float);reference=np.asarray(reference_xy);quality=np.maximum(np.asarray(quality),.01);origins=np.asarray(origins);steps=np.asarray(steps)
    if len(reference)<10:return None,dict(reason='too_few_observations')
    dt=np.diff(times);speed=np.linalg.norm(np.diff(reference,axis=0),axis=1)/dt;speed=speed[dt<=2.]
    median_speed=float(np.median(speed)) if len(speed) else float('inf')
    if median_speed>.3:return None,dict(reason='moving_track',median_reference_speed_m_s=median_speed)
    unit=steps/np.linalg.norm(steps,axis=1)[:,None];projector=np.eye(2)[None]-unit[:,:,None]*unit[:,None,:];centre=np.median(reference,axis=0)
    for _ in range(30):
        residual=np.linalg.norm(np.einsum('nij,nj->ni',projector,centre-origins),axis=1);weights=quality/np.sqrt(1+(residual/.2)**2);matrix=np.einsum('n,nij->ij',weights,projector);rhs=np.einsum('n,nij,nj->i',weights,projector,origins);centre=np.linalg.solve(matrix+np.eye(2)*1e-9,rhs)
    residual=np.linalg.norm(np.einsum('nij,nj->ni',projector,centre-origins),axis=1);fraction=float((residual<=.5).mean());metric_fraction=float(np.sum(quality*(np.linalg.norm(reference-centre,axis=1)<=1.))/np.sum(quality));condition=float(np.linalg.cond(matrix));diag=dict(median_reference_speed_m_s=median_speed,camera_ray_inlier_fraction=fraction,metric_consensus_fraction=metric_fraction,camera_ray_matrix_condition=condition)
    if fraction<.9 or metric_fraction<.7 or condition>100:return None,dict(reason='stationary_centre_not_observable',**diag)
    return centre,dict(reason='stationary_supported_by_motion_and_rays',centre_xy_map=centre.tolist(),**diag)
