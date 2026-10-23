"""Independent saved-output checks and per-ID review sheet (no label mutation)."""
import argparse,json
from pathlib import Path
import numpy as np
from person_linear_recovery import find_outliers


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/zgc_odom_local.json');args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    root=Path(config['temporal_output']);video=root/'map_review/sam3_full_video';out=video/'interior_half_review/consistent_tracks'
    with np.load(out/'centers_only.npz') as z:data={k:z[k] for k in z.files}
    with np.load(config['pose_npz']) as z:poses=z['transforms']
    presence=json.loads((video/'visual_presence_index.json').read_text());report=json.loads((out/'report.json').read_text())
    good=data['position_available'];xy=data['corrected_xy_map'];ids=data['sam_video_ids'];frames=data['frame_ids'];times=data['timestamp_ns']*1e-9
    assert np.isfinite(xy[good]).all()
    assert len(set(zip(frames, data['instance_ids'])))==len(frames)
    errors=[];world_errors=[];depths=[];source=Path(config['source_index']).parent/'frames'
    for f in np.unique(frames[good]):
        with np.load(source/f'{f:06d}.npz') as z:E=z['cam_extrinsic'];K=z['cam_intrinsic']
        ix=np.flatnonzero(good&(frames==f));old=data['original_point_lidar'][ix]@E[:3,:3].T+E[:3,3];new=data['corrected_point_lidar'][ix]@E[:3,:3].T+E[:3,3]
        a=old@K.T;b=new@K.T;errors.extend(np.linalg.norm(a[:,:2]/a[:,2:]-b[:,:2]/b[:,2:],axis=1));depths.extend(new[:,2])
        world=data['corrected_point_lidar'][ix]@poses[f,:3,:3].T+poses[f,:3,3];world_errors.extend(np.linalg.norm(world[:,:2]-xy[ix],axis=1))
    flagged=[];jumps=[];tracks=[]
    for tid in np.unique(ids[good]):
        ix=np.flatnonzero(good&(ids==tid));bad,_=find_outliers(frames[ix],times[ix],xy[ix],set(presence[str(tid)]))
        flagged.extend(dict(id=int(tid),frame=int(frames[i])) for i in ix[bad])
        dt=np.diff(times[ix]);distance=np.linalg.norm(np.diff(xy[ix],axis=0),axis=1)
        jumps.extend(dict(id=int(tid),frame=int(frames[ix[k+1]])) for k in np.flatnonzero((dt<=2)&(distance>3*dt+.350001)))
        tracks.append(dict(id=int(tid),retained=len(ix)))
    residual=np.linalg.norm(xy[good]-data['reference_xy_map'][good],axis=1)
    retained=int(good.sum())
    result=dict(population='all saved observations; no subsampling',input=len(frames),retained=retained,ignored=int((~good).sum()),max_pixel_change_px=float(max(errors)) if retained else None,min_camera_depth_m=float(min(depths)) if retained else None,max_world_transform_error_m=float(max(world_errors)) if retained else None,max_reference_distance_m=float(max(residual)) if retained else None,independent_local_linear_outliers=flagged,adjacent_jump_outliers=jumps,tracks=tracks,nonconverged_reference_models=[m for m in report['reference_models'] if not m['converged']],labels_modified=False)
    (out/'integrity_audit.json').write_text(json.dumps(result,indent=2))
    if retained:
        assert max(errors)<1e-7 and min(depths)>.1 and max(world_errors)<1e-7 and max(residual)<.500001
        assert not flagged and not jumps,result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background');fig,axes=plt.subplots(4,5,figsize=(20,16),dpi=130)
    for ax,tid in zip(axes.flat,np.unique(ids[good])):
        ix=np.flatnonzero(good&(ids==tid));p=xy[ix];ref=data['reference_xy_map'][ix]
        cuts=np.r_[0,np.flatnonzero(np.diff(times[ix])>2.)+1,len(ix)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            ax.plot(ref[lo:hi,1],ref[lo:hi,0],color='#ffe16a',lw=.8,alpha=.7,label='Reference')
        ax.scatter(p[:,1],p[:,0],c='#00ffff',s=12,label='Final centre')
        extent=np.ptp(p,axis=0);span=max(float(extent.max())*1.15,1.2);centre=(p.max(0)+p.min(0))/2
        ax.set_xlim(centre[1]+span/2,centre[1]-span/2);ax.set_ylim(centre[0]-span/2,centre[0]+span/2);ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.set_title(f'P{tid} | {len(ix)} positions');ax.set_xlabel('Map Y (m)');ax.set_ylabel('Map X (m)')
    for ax in list(axes.flat)[len(tracks):]:ax.axis('off')
    fig.suptitle('All retained pedestrian IDs | cyan: one centre per frame | yellow: reference\nCamera ray fixed; walking tracks remain moving; review candidates, not verified ground truth',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'all_ids_review.png');plt.close(fig)
    page=out/'index.html';html=page.read_text();link='<p><a href="integrity_audit.json">独立全量复核</a></p><img src="all_ids_review.png">'
    if 'all_ids_review.png' not in html:page.write_text(html+link)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
