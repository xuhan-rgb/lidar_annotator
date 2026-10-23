"""WildDet3D person cuboid conversion and camera-ray depth-only refinement."""
import numpy as np
from scipy.optimize import minimize_scalar

# Native WildDet3D cuboid-corner index -> the verified Cube R-CNN cuboid basis.
NATIVE_TO_CUBE_ORDER=np.array([2,0,4,6,3,1,5,7],dtype=np.int64)
BOX_SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],
                    [-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]],dtype=float)


def box_to_corners(box):
    """Decode the stored upright [x,y,z,L,W,H,yaw] LiDAR convention."""
    box=np.asarray(box,dtype=float)
    if box.shape!=(7,) or not np.isfinite(box).all() or np.any(box[3:6]<=0):
        raise ValueError('Invalid upright person box')
    c,s=np.cos(box[6]),np.sin(box[6])
    axes=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return box[:3]+(BOX_SIGNS*(box[3:6]/2))@axes.T


def huber(values,delta=.15):
    a=np.abs(np.asarray(values,dtype=float))
    return np.where(a<=delta,.5*a*a,delta*(a-.5*delta))


def person_box_geometry(item,lidar_to_camera,center_tolerance=2e-5):
    """Return camera/LiDAR center, physical LWH/yaw, corners, axes and fixed offsets."""
    record=np.asarray(item['box_camera_center_wlh_quaternion'],dtype=float)
    native=np.asarray(item['corners_cam'],dtype=float)
    E=np.asarray(lidar_to_camera,dtype=float)
    if record.shape!=(10,) or native.shape!=(8,3) or E.shape!=(4,4):
        raise ValueError('Invalid WildDet3D cuboid/calibration shape')
    if not np.isfinite(record).all() or not np.isfinite(native).all() or not np.isfinite(E).all():
        raise ValueError('Non-finite WildDet3D cuboid/calibration')
    center_camera=record[:3]
    width,length,height=map(float,record[3:6])
    if min(width,length,height)<=0 or center_camera[2]<=.1:
        raise ValueError('Invalid person cuboid dimensions or depth')
    corners_camera=native[NATIVE_TO_CUBE_ORDER]
    if not np.allclose(corners_camera.mean(0),center_camera,atol=center_tolerance):
        raise ValueError('WildDet3D corners and center disagree')
    camera_to_lidar=np.linalg.inv(E)
    corners_lidar=corners_camera@camera_to_lidar[:3,:3].T+camera_to_lidar[:3,3]
    center_lidar=center_camera@camera_to_lidar[:3,:3].T+camera_to_lidar[:3,3]
    if not np.allclose(corners_lidar.mean(0),center_lidar,atol=center_tolerance):
        raise ValueError('Transformed corners and center disagree')
    # Preserve only the predicted long-axis azimuth. Raw Mono3D roll/pitch must
    # not leak into an output box that stores a single LiDAR yaw value.
    eL=corners_lidar[1]-corners_lidar[0]
    horizontal_length=float(np.linalg.norm(eL[:2]))
    if horizontal_length<1e-5:raise ValueError('WildDet3D long axis has no horizontal component')
    yaw=float(np.arctan2(eL[1],eL[0])%np.pi)
    c,s=np.cos(yaw),np.sin(yaw)
    axes=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    dims=np.array([length,width,height],float)
    box=np.array([*center_lidar.tolist(),*dims.tolist(),yaw],dtype=float)
    corners_lidar=box_to_corners(box)
    return dict(box=box,center_camera=center_camera,center_lidar=center_lidar,
        corners_camera=corners_camera,corners_lidar=corners_lidar,axes=axes,
        offsets=corners_lidar-center_lidar,half=dims/2,camera_to_lidar=camera_to_lidar,
        dimensions_wilddet_wlh=[width,length,height])


def box_metrics(center,axes,half,points):
    """Compute signed cuboid-surface distances and first-ray entry error in LiDAR frame."""
    center=np.asarray(center,float);axes=np.asarray(axes,float);half=np.asarray(half,float);points=np.asarray(points,float)
    local=(points-center)@axes
    q=np.abs(local)-half
    outside=np.linalg.norm(np.maximum(q,0),axis=1)
    surface=outside+np.minimum(q.max(1),0)
    distances=np.linalg.norm(points,axis=1)
    directions=(points/np.maximum(distances[:,None],1e-8))@axes
    origin=(-center)@axes
    near=np.full(len(points),-np.inf);far=np.full(len(points),np.inf);possible=np.ones(len(points),bool)
    for dim in range(3):
        parallel=np.abs(directions[:,dim])<1e-10
        possible &= ~parallel | (abs(origin[dim])<=half[dim])
        divisor=np.where(parallel,1.,directions[:,dim])
        t1=(-half[dim]-origin[dim])/divisor;t2=(half[dim]-origin[dim])/divisor
        near=np.maximum(near,np.where(parallel,-np.inf,np.minimum(t1,t2)))
        far=np.minimum(far,np.where(parallel,np.inf,np.maximum(t1,t2)))
    hit=possible & (far>=np.maximum(near,0))
    ray_error=np.where(hit,np.maximum(near,0)-distances,outside)
    return surface,ray_error,outside,hit


def refine_depth(geometry,points,delta=.15):
    """Optimize center Z along WildDet3D camera-center ray; preserve predicted shape/orientation."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<5 or not np.isfinite(points).all():
        return dict(success=False,reason='insufficient_or_invalid_person_support')
    cam=np.asarray(geometry['center_camera'],float);z0=float(cam[2]);ray=cam/z0
    inv=geometry['camera_to_lidar'];axes=geometry['axes'];half=geometry['half'];offsets=geometry['offsets']
    def at(z):return (ray*float(z))@inv[:3,:3].T+inv[:3,3]
    def objective(z):
        center=at(z);surface,ray_error,_,_=box_metrics(center,axes,half,points)
        return float(huber(surface,delta).mean()+huber(ray_error,delta).mean())
    upper=max(2*z0,float(np.quantile(np.linalg.norm(points,axis=1),.95)+np.linalg.norm(half)))
    if not np.isfinite(upper) or upper<=.2:return dict(success=False,reason='invalid_depth_bounds')
    grid=np.linspace(.2,upper,241);costs=np.asarray([objective(z) for z in grid]);best=int(costs.argmin())
    lo=grid[max(0,best-1)];hi=grid[min(len(grid)-1,best+1)]
    if hi<=lo:lo,hi=.2,upper
    fit=minimize_scalar(objective,bounds=(float(lo),float(hi)),method='bounded',options={'xatol':1e-7})
    center=at(float(fit.x));surface,ray_error,outside,hit=box_metrics(center,axes,half,points)
    corners=offsets+center;box=np.array([*center.tolist(),*geometry['box'][3:]],dtype=float)
    return dict(success=bool(fit.success),box_3d=box,corners_lidar=corners,center_lidar=center,
        depth_camera_z_m=float(fit.x),depth_delta_camera_m=float(fit.x-z0),objective=float(fit.fun),
        objective_before=objective(z0),median_surface_distance_m=float(np.median(np.abs(surface))),
        median_ray_error_m=float(np.median(np.abs(ray_error))),ray_hit_count=int(hit.sum()),
        support_inside_count=int((outside<=1e-5).sum()),optimizer_message=str(fit.message))
