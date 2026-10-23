"""Geometric camera frustum in a LiDAR height slab; never a background label."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .display_range import DEFAULT_LIMITS


def camera_coverage(intrinsic,extrinsic,image_size,xyz,limits=DEFAULT_LIMITS,resolution=.25):
    K=np.asarray(intrinsic,float);E=np.asarray(extrinsic,float);extent,zmin,zmax=limits;width,height=image_size
    if not np.isfinite(K).all() or not np.isfinite(E).all():raise ValueError('相机标定包含非有限值')
    if not np.allclose(K[2],[0,0,1]):raise ValueError('相机内参最后一行必须为 [0,0,1]')
    if extent<=0 or zmin>=zmax or resolution<=0 or width<=0 or height<=0:raise ValueError('视觉范围参数无效')
    cells=int(np.ceil(2*extent/resolution));step=2*extent/cells
    axis=-extent+(np.arange(cells)+.5)*step;x,y=np.meshgrid(axis,axis)
    P=K@E[:3];depth=E[2].copy();depth[3]-=1e-6
    # Each image/depth half-space gives a linear bound on LiDAR Z at fixed XY.
    planes=np.array([depth,P[0],width*P[2]-P[0],P[1],height*P[2]-P[1]])
    lower=np.full(x.shape,zmin);upper=np.full(x.shape,zmax);valid=np.ones(x.shape,bool)
    for a,b,c,d in planes:
        base=a*x+b*y+d
        if abs(c)<1e-12:valid&=base>=0
        elif c>0:lower=np.maximum(lower,-base/c)
        else:upper=np.minimum(upper,-base/c)
    mask=valid&(lower<=upper)
    camera=np.asarray(xyz)@E[:3,:3].T+E[:3,3];pixels=camera@K.T
    point_mask=np.isfinite(camera).all(1)&(camera[:,2]>1e-6)
    uv=np.zeros((len(camera),2));uv[point_mask]=pixels[point_mask,:2]/pixels[point_mask,2:3]
    point_mask&=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    # Raster row is +Y, column is +X. Edges are in metric LiDAR XY.
    padded=np.pad(mask,1);segments=[]
    for neighbor,offset,direction in [(padded[:-2,1:-1],(0,0),(1,0)),(padded[2:,1:-1],(0,1),(1,0)),(padded[1:-1,:-2],(0,0),(0,1)),(padded[1:-1,2:],(1,0),(0,1))]:
        rows,cols=np.nonzero(mask&~neighbor)
        start=np.c_[cols,rows]+offset
        segments.append(np.stack([start,start+direction],axis=1)*step-extent)
    return dict(bev_fov_mask=mask,point_fov_mask=point_mask,boundary_xy=np.concatenate(segments).astype(np.float32),
                xy_min=np.array([-extent,-extent]),resolution=np.asarray(step),z_range=np.array([zmin,zmax]),image_size=np.array([width,height]),
                cam_intrinsic=K,cam_extrinsic=E)


def save_coverage(directory,result,report):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # Per-point visibility depends on each scan, so it is not part of a common mask.
    arrays={key:value for key,value in result.items() if key!='point_fov_mask'}
    arrays['metadata']=np.asarray(json.dumps(report,ensure_ascii=False))
    temporary=directory/'.visibility_mask.tmp.npz';np.savez_compressed(temporary,**arrays);temporary.replace(directory/'visibility_mask.npz')
    previous=directory/'index.json'
    if previous.is_file():
        old=json.loads(previous.read_text())
        if old.get('schema')=='camera_geometric_coverage_v1':
            for row in old['rows']:
                for key in ('mask','archive'):
                    name=row[key];path=Path(name)
                    if path.name==name and len(path.stem)==6 and path.stem.isdigit() and path.suffix in ('.png','.npz'):
                        (directory/name).unlink(missing_ok=True)
            previous.unlink()


def boundary_curves(segments):
    neighbors={}
    for i,(start,end) in enumerate(segments):
        a,b=tuple(start),tuple(end)
        neighbors.setdefault(a,[]).append((i,b));neighbors.setdefault(b,[]).append((i,a))
    used=set();curves=[]
    for i,(start,_) in enumerate(segments):
        if i in used:continue
        point=tuple(start);curve=[point]
        while True:
            edges=neighbors[point]
            while edges and edges[-1][0] in used:edges.pop()
            if not edges:break
            edge,point=edges.pop();used.add(edge);curve.append(point)
        curves.append(np.asarray(curve))
    return curves


def generate_coverage(job):
    job=Path(job);raw=job/'raw';rows=json.loads((raw/'index.json').read_text())['rows'];destination=job/'visibility_masks'
    owner=json.loads((job/'job.json').read_text());directories=[destination];reference=None
    if owner.get('dataset_output'):directories.append(Path(owner['dataset_output'])/'labels/visibility_masks')
    for row in rows:
        with np.load(raw/row['path'],allow_pickle=False) as z:
            K=z['cam_intrinsic'];E=z['cam_extrinsic']
            with Image.open(row['image_path']) as image:size=image.size
            if reference is None:reference=(K.copy(),E.copy(),size)
            elif size!=reference[2] or not np.array_equal(K,reference[0]) or not np.array_equal(E,reference[1]):
                raise ValueError('各帧相机标定或图像尺寸不同，不能共用一个视觉范围 mask')
    if reference is None:raise ValueError('没有可生成视觉范围 mask 的帧')
    result=camera_coverage(*reference,np.empty((0,3)))
    report=dict(schema='camera_geometric_coverage_v2',frames=len(rows),shared_across_frames=True,xy_range=[-20,20],z_range=[-3,6],resolution=.25,
                raster_axes='row increases with LiDAR Y; column increases with LiDAR X; origin [-20,-20] at lower cell edge',
                meaning='1: cell centre XY intersects camera frustum at some Z in the height slab; 0: outside geometric coverage',
                occlusion_checked=False,negative_supervision=False,labels_modified=False)
    for directory in directories:save_coverage(directory,result,report)
    print(f'通用视觉范围 mask：{destination/"visibility_mask.npz"}（{len(rows)} 帧共用；仅几何覆盖，不代表完整标注）',flush=True)
    return report
