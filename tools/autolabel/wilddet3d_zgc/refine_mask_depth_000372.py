"""One-variable camera-ray translation using mask-selected LiDAR returns."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,MODEL_CACHE,PROJECT_ROOT
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize_scalar


ROOT=DATA/'wilddet3d_image_test_20261013'
OUT=ROOT/'mask_lidar_depth_000372'

def box_metrics(z,ray,inverse,offsets,axes,half,points):
    center=(z*ray)@inverse[:3,:3].T+inverse[:3,3]
    local=(points-center)@axes
    q=np.abs(local)-half
    outside=np.linalg.norm(np.maximum(q,0),axis=1)
    surface=outside+np.minimum(q.max(1),0)
    # Rays originate at the LiDAR optical origin, not the camera optical origin.
    distance=np.linalg.norm(points,axis=1)
    directions=(points/distance[:,None])@axes
    origin=(-center)@axes
    near=np.full(len(points),-np.inf);far=np.full(len(points),np.inf)
    possible=np.ones(len(points),bool)
    for dim in range(3):
        parallel=np.abs(directions[:,dim])<1e-10
        possible&=~parallel | (abs(origin[dim])<=half[dim])
        divisor=np.where(parallel,1,directions[:,dim])
        t1=(-half[dim]-origin[dim])/divisor;t2=(half[dim]-origin[dim])/divisor
        near=np.maximum(near,np.where(parallel,-np.inf,np.minimum(t1,t2)))
        far=np.minimum(far,np.where(parallel,np.inf,np.maximum(t1,t2)))
    hit=possible&(far>=np.maximum(near,0))
    ray_error=np.where(hit,np.maximum(near,0)-distance,outside)
    return center,surface,ray_error,outside,hit

def huber(values,delta=.15):
    a=np.abs(values)
    return np.where(a<=delta,.5*a*a,delta*(a-.5*delta))

def main():
    OUT.mkdir(exist_ok=True)
    source=DATA/'labels/zgc'
    with np.load(source/'frames/000372.npz') as a:
        xyz=a['xyz'].astype(float);uv=a['pixels_uv'];visible=a['visible'];depth=a['camera_depth'];K=a['cam_intrinsic'];E=a['cam_extrinsic']
    meta=json.loads((source/'metadata/000372.json').read_text());mask=np.asarray(Image.open(source/meta['mask']))
    prediction=json.loads((ROOT/'text_car_000372.json').read_text())['items'][0]
    bbox=np.asarray(prediction['bbox_xyxy'])
    valid=visible&np.isfinite(xyz).all(1)&np.isfinite(uv).all(1)&(depth>.1)
    ids=np.flatnonzero(valid);pixels=np.floor(uv[ids]).astype(int)
    valid[ids]&=(pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]<mask.shape[0])
    ids=np.flatnonzero(valid);pixels=np.floor(uv[ids]).astype(int)
    semantic=np.full(len(xyz),255,np.uint8);semantic[ids]=mask[pixels[:,1],pixels[:,0]]
    ids=np.flatnonzero(valid&(semantic==2)&(uv>=bbox[:2]).all(1)&(uv<bbox[2:]).all(1))
    groups=DBSCAN(eps=.4,min_samples=3).fit_predict(xyz[ids]);counts=[(int((groups==g).sum()),int(g)) for g in np.unique(groups) if g>=0]
    count,group=max(counts);selected=ids[groups==group];points=xyz[selected]
    inverse=np.linalg.inv(E);original_camera=np.asarray(prediction['box_camera_center_wlh_quaternion'][:3]);initial_z=float(original_camera[2]);ray=original_camera/initial_z
    camera_corners=np.asarray(prediction['corners_cam']);lidar_corners=camera_corners@inverse[:3,:3].T+inverse[:3,3]
    original_center=original_camera@inverse[:3,:3].T+inverse[:3,3];offsets=lidar_corners-original_center
    w,l,h=prediction['box_camera_center_wlh_quaternion'][3:6]
    axes=np.column_stack(((lidar_corners[0]-lidar_corners[2])/l,(lidar_corners[1]-lidar_corners[0])/w,(lidar_corners[0]-lidar_corners[4])/h));half=np.array([l,w,h])/2
    np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=2e-5)
    args=(ray,inverse,offsets,axes,half,points)
    def objective(z):
        _,surface,ray_error,_,_=box_metrics(z,*args)
        return float(np.mean(huber(surface))+np.mean(huber(ray_error)))
    bounds=(.2,max(2*initial_z,float(np.quantile(depth[selected],.95)+np.linalg.norm(half))))
    # Coarse objective search locates the minimum's bracket; continuous refinement follows.
    grid=np.linspace(*bounds,241);costs=np.array([objective(z) for z in grid]);best=int(costs.argmin())
    lower=grid[max(0,best-1)];upper=grid[min(len(grid)-1,best+1)]
    fit=minimize_scalar(objective,bounds=(lower,upper),method='bounded',options={'xatol':1e-7})
    def describe(z):
        center,surface,ray_error,outside,hit=box_metrics(z,*args);corners=offsets+center
        camera=corners@E[:3,:3].T+E[:3,3];projected=camera@K.T;projected=projected[:,:2]/projected[:,2,None]
        center_cam=center@E[:3,:3].T+E[:3,3];center_uv=K@center_cam;center_uv=center_uv[:2]/center_uv[2]
        all_local=(xyz-center)@axes;all_support=(np.abs(all_local)<=half+1e-5).all(1)
        return {'depth_camera_z_m':float(z),'center_lidar_xyz':center.tolist(),'center_uv':center_uv.tolist(),'corners_lidar':corners.tolist(),'corners_camera':camera.tolist(),'corners_uv':projected.tolist(),'objective':objective(z),'median_surface_distance_m':float(np.median(np.abs(surface))),'median_ray_error_m':float(np.median(np.abs(ray_error))),'mask_points_inside_box':int((outside<1e-5).sum()),'mask_points_within_015m_of_surface':int((np.abs(surface)<=.15).sum()),'mask_ray_hit_count':int(hit.sum()),'all_lidar_points_inside_box':int(all_support.sum())}
    before=describe(initial_z);after=describe(fit.x)
    np.testing.assert_allclose(before['center_uv'],after['center_uv'],atol=1e-6)
    origin=inverse[:3,3];direction=original_center-origin;delta=np.array(after['center_lidar_xyz'])-origin
    assert np.linalg.norm(np.cross(direction,delta))<1e-6
    np.testing.assert_allclose(np.asarray(after['corners_lidar'])-np.asarray(after['center_lidar_xyz']),offsets,atol=1e-6)
    report={'frame':'000372','prediction_index':0,'status':'complete','optimized_parameter':'Camera center Z only; center follows the camera optical ray in LiDAR coordinates','fixed_parameters':['image projected center','dimensions','rotation'],'objective':'Mean Huber distance to cuboid surface + mean Huber LiDAR first-ray-entry error; both weights1, delta0.15m','mask':str(source/meta['mask']),'raw_mask_bbox_points':len(ids),'selected_points':len(selected),'point_selection':'All visible vehicle-mask pixels inside predicted2Dbox, then largest3DDBSCAN cluster eps0.4m/min_samples3; no random or first-N subsampling','cluster_counts':counts,'camera_origin_lidar':origin.tolist(),'bounds_camera_z_m':list(bounds),'optimizer_success':bool(fit.success),'before':before,'after':after,'projection_center_and_rigid_shape_checks':'passed','limitations':'Single frame. A cuboid approximates curved car surfaces. Fixed center ray, size and orientation may themselves be wrong. This is optimization of observed point consistency, not a ground-truth accuracy measurement. Original predictions/masks/training labels remain unchanged.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    np.savez_compressed(OUT/'000372_depth_refinement.npz',selected_point_indices=selected,selected_lidar_points=points,selected_pixels_uv=uv[selected],depth_grid=grid,objective_grid=costs,corners_before=np.array(before['corners_lidar']),corners_after=np.array(after['corners_lidar']),camera_K=K,lidar_to_camera=E)
    print(json.dumps({'selected_points':len(selected),'before':{k:before[k] for k in ['depth_camera_z_m','center_lidar_xyz','objective','median_surface_distance_m','median_ray_error_m','mask_points_inside_box']},'after':{k:after[k] for k in ['depth_camera_z_m','center_lidar_xyz','objective','median_surface_distance_m','median_ray_error_m','mask_points_inside_box']},'optimizer_success':fit.success},indent=2))
if __name__=='__main__':main()
