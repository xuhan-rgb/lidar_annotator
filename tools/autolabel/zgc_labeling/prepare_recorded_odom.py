"""Prepare local pose matrices from recorded [x,y,z,yaw]; no synthetic IMU or ICP resets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def pose_matrices(poses):
    poses=np.asarray(poses,float)
    if poses.ndim!=2 or poses.shape[1]!=4 or not np.isfinite(poses).all():raise ValueError('Expected finite [x,y,z,yaw] poses')
    transforms=np.repeat(np.eye(4)[None],len(poses),axis=0)
    c,s=np.cos(poses[:,3]),np.sin(poses[:,3]);transforms[:,0,0]=c;transforms[:,0,1]=-s;transforms[:,1,0]=s;transforms[:,1,1]=c;transforms[:,:3,3]=poses[:,:3]
    return np.linalg.inv(transforms[0])[None]@transforms


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args();config=json.loads(args.config.read_text())
    output=Path(config['source_index']).parent.parent.parent/'zgc_odom_local_refined_20261014';output.mkdir(parents=True,exist_ok=True)
    with np.load(config['odometry_npz']) as z:poses=z['poses'];times=z['timestamps']
    transforms=pose_matrices(poses);segments=np.zeros(len(poses),np.int32);diagnostics=[]
    for i in range(len(poses)):
        valid=True
        if i:
            dt=(int(times[i])-int(times[i-1]))/1e9
            step=np.linalg.norm(poses[i,:3]-poses[i-1,:3]);angle=abs((poses[i,3]-poses[i-1,3]+np.pi)%(2*np.pi)-np.pi)
            valid=0<dt<=.5 and step<=1. and angle<=np.deg2rad(10)
            segments[i]=segments[i-1]+int(not valid)
        diagnostics.append(dict(frame=f'{i:06d}',valid=bool(valid),source='recorded odom; no ICP fitness gate'))
    np.savez_compressed(output/'poses.npz',transforms=transforms,segments=segments)
    assumption='Stored cloud is interpreted in the same gravity-aligned current frame as recorded xyz/yaw; identity LiDAR/body transform assumed, not independently calibrated here. Roll/pitch are not available in this cached pose file.'
    report=dict(method='recorded odom Rz(yaw)+translation, rebased to first frame',frames=len(poses),segments=int(segments[-1])+1,diagnostics=diagnostics,coordinate_assumption=assumption,source_sha256=hashlib.sha256(Path(config['odometry_npz']).read_bytes()).hexdigest())
    (output/'pose_report.json').write_text(json.dumps(report,indent=2))
    config.update(temporal_output=str(output),pose_npz=str(output/'poses.npz'),pose_report=str(output/'pose_report.json'),vehicle_strategy='local_odom_anchor',pose_source='recorded_odom',coordinate_assumption=assumption)
    config['temporal_parameters']={**config.get('temporal_parameters',{}),'local_anchor_max_age_s':2.}
    dest=Path(__file__).parent/'configs/zgc_odom_local.json';dest.write_text(json.dumps(config,indent=2));print(json.dumps(dict(config=str(dest),frames=len(poses),segments=report['segments']),indent=2))


if __name__=='__main__':main()
