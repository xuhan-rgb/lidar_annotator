"""Review-only static track fits: fixed anchor camera ray/size, robust history edges."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anchor_vehicle_refine import select_anchor
from temporal_refine import world_heading,wrap_angle


def edge_distances(points,center,yaw,size,origins):
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    q=(points[:,:2]-center[:2])@R; v=(origins[:,:2]-center[:2])@R; half=size[:2]/2
    distances=[]
    for axis in (0,1):
        normal=q[:,axis]-np.sign(v[:,axis])*half[axis]
        tangent=np.maximum(np.abs(q[:,1-axis])-half[1-axis],0.)
        distances.append(np.where(np.abs(v[:,axis])>half[axis],np.hypot(normal,tangent),10.))
    return np.minimum(*distances)


def fit_history(points,origins,frames,box,origin,step,depth,raw_yaw):
    _,inverse,counts=np.unique(frames,return_inverse=True,return_counts=True)
    weight=1/np.sqrt(counts[inverse])
    yaw0=box[6]; raw=yaw0-wrap_angle(yaw0-raw_yaw)
    lower=max(yaw0-np.pi/2,raw-np.pi/2);upper=min(yaw0+np.pi/2,raw+np.pi/2)
    def decode(x):
        b=box.copy();b[:3]=origin+x[0]*step;b[6]=x[1];return b
    def residual(x):
        b=decode(x);d=edge_distances(points,b[:3],b[6],b[3:6],origins)
        # Mean robust edge cost per frame; dense frames cannot dominate by point count.
        robust=np.sqrt(2*.15**2*(np.sqrt(1+(d/.15)**2)-1))*weight
        return np.r_[robust,.05*(x[0]-depth),.05*wrap_angle(x[1]-yaw0)]
    bounds=([max(.2,depth-2),lower],[depth+2,upper])
    initial=np.array([depth,np.clip(yaw0,lower+1e-7,upper-1e-7)])
    candidates=[least_squares(residual,[depth,np.clip(yaw0+delta,lower+1e-7,upper-1e-7)],bounds=bounds,max_nfev=60) for delta in [-.5,0,.5]]
    fit=min(candidates,key=lambda f:np.dot(f.fun,f.fun));before=np.dot(residual(initial),residual(initial))
    return (decode(fit.x) if fit.success and np.dot(fit.fun,fit.fun)<=before else box.copy()),dict(converged=bool(fit.success),objective_before=float(before),objective_after=float(np.dot(fit.fun,fit.fun)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=json.loads(args.config.read_text());root=Path(c['temporal_output']);out=root/'map_review'/'history_fit';out.mkdir(exist_ok=True)
    rows=json.loads(Path(c['source_index']).read_text())['rows'];source=Path(c['source_index']).parent
    with np.load(c['pose_npz']) as z:poses=z['transforms']
    groups=defaultdict(list)
    for row,T in zip(rows,poses):
        f=int(Path(row['path']).stem);meta=json.loads((root/'metadata'/f'{f:06d}.json').read_text())
        cars=[o for o in meta['observations'] if o['kind']=='vehicle' and o.get('track_id') is not None and o['status'] in ['accepted_temporal','accepted_single_frame']]
        if not cars:continue
        with np.load(source/row['path']) as z:xyz=z['xyz'].astype(float);K=z['cam_intrinsic'];E=z['cam_extrinsic']
        for o in cars:
            o.update(box=np.array(o['box']),points=xyz[o['support_ids']],T=T,K=K,E=E);groups[o['track_id']].append(o)
    results=[]
    with np.load(root/'map_review/map_points.npz') as z:bg=z['background_xyz_map']
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(17,13),dpi=140)
    ax.scatter(bg[:,1],bg[:,0],s=.15,c='#87939c',alpha=.3,linewidths=0)
    for tid,obs in sorted(groups.items()):
        chosen=select_anchor(obs)
        if chosen is None:results.append(dict(track_id=tid,status='no_reliable_anchor'));continue
        anchor,quality=chosen;T=anchor['T'];box=anchor['box'].copy();box[:3]=T[:3,:3]@box[:3]+T[:3,3];box[6]=world_heading(box[6],T)
        points=np.concatenate([o['points']@o['T'][:3,:3].T+o['T'][:3,3] for o in obs]);frames=np.concatenate([np.full(len(o['points']),o['frame']) for o in obs]);origins=np.concatenate([np.tile(o['T'][:3,3],(len(o['points']),1)) for o in obs])
        inv=np.linalg.inv(anchor['E']);cam=anchor['E'][:3,:3]@anchor['box'][:3]+anchor['E'][:3,3];depth=cam[2]
        origin=T[:3,:3]@inv[:3,3]+T[:3,3];step=T[:3,:3]@inv[:3,:3]@(cam/depth)
        raw=world_heading(anchor['initial_yaw'],T);new,diag=fit_history(points,origins,frames,box,origin,step,depth,raw)
        assert abs(wrap_angle(new[6]-raw))<=np.pi/2+1e-6
        assert np.linalg.norm(np.cross(new[:3]-origin,step))<1e-6
        before=edge_distances(points,box[:3],box[6],box[3:6],origins);after=edge_distances(points,new[:3],new[6],new[3:6],origins)
        results.append(dict(track_id=tid,status='review_candidate',anchor_frame=anchor['frame'],frames=len(obs),points=len(points),anchor_quality=quality,initial_box_map=box.tolist(),candidate_box_map=new.tolist(),edge_median_before_m=float(np.median(before)),edge_median_after_m=float(np.median(after)),yaw_change_deg=float(np.degrees(wrap_angle(new[6]-box[6]))),center_shift_m=float(np.linalg.norm(new[:3]-box[:3])),**diag))
        ax.scatter(points[:,1],points[:,0],s=.4,c='#89d5ed',alpha=.3,linewidths=0)
        for b,color,style in [(box,'#fb7185','--'),(new,'#22e690','-')]:
            q=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*b[3:5]/2;R=np.array([[np.cos(b[6]),-np.sin(b[6])],[np.sin(b[6]),np.cos(b[6])]]);q=q@R.T+b[:2];ax.plot(q[:,1],q[:,0],style,c=color,lw=1.3)
        ax.text(new[1],new[0],f'T{tid}',fontsize=9,color='#fff')
        if tid==8:
            fig2,a=plt.subplots(figsize=(10,9),dpi=140);im=a.scatter(points[:,1],points[:,0],c=frames,cmap='turbo',s=2,alpha=.5);fig2.colorbar(im,ax=a,label='Source frame ID')
            for b,col,style,label in [(box,'#fb7185','--','Best anchor before'),(new,'#22e690','-','History optimized candidate')]:
                q=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*b[3:5]/2;R=np.array([[np.cos(b[6]),-np.sin(b[6])],[np.sin(b[6]),np.cos(b[6])]]);q=q@R.T+b[:2];a.plot(q[:,1],q[:,0],style,c=col,lw=2,label=label)
            a.set_aspect('equal');a.invert_xaxis();a.set_xlabel('Map Y (m), +Y left');a.set_ylabel('Map X (m)');a.legend();a.grid(alpha=.2);a.set_title('T8: historical points / fixed anchor pixel ray / depth + yaw only');fig2.tight_layout();fig2.savefig(out/'T008_before_after.png');plt.close(fig2)
        print(f'T{tid}: median {np.median(before):.3f} -> {np.median(after):.3f} m',flush=True)
    ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m)');ax.grid(alpha=.15);ax.set_title('History fit REVIEW · pink dashed: best anchor / green: optimized candidate\nRecorded odom fixed; track IDs not merged; labels unchanged');fig.tight_layout();fig.savefig(out/'whole_map_before_after.png');plt.close(fig)
    (out/'report.json').write_text(json.dumps(dict(labels_modified=False,poses_modified=False,scope='per existing track, accepted observations only; diagnostic map candidates, not exported labels',objective='sum of per-frame mean soft-L1 visible-edge cost (delta .15m) + depth/yaw priors coefficient .05; fixed anchor size and camera ray; depth bound +/-2m; yaw within90deg of anchor raw heading',results=results),indent=2))


if __name__=='__main__':main()
