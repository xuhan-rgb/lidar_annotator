"""Robust whole-track reference and camera-ray-constrained centres, review only."""
import argparse,json,time,colorsys
from pathlib import Path
import numpy as np
from person_track_reference import robust_reference,depth_on_reference_tube,static_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['temporal_output']);base=root/'map_review/sam3_full_video/interior_half_review';out=base/'consistent_tracks';out.mkdir(exist_ok=True);(out/'metadata').mkdir(exist_ok=True);source=Path(c['source_index']).parent
    with np.load(base/'map_centers_only.npz') as z:raw=z['xy_map'];frames=z['frame_ids'];instances=z['instance_ids'];ids=z['sam_video_ids']
    with np.load(c['pose_npz']) as z:poses=z['transforms']
    audit=json.loads((root/'map_review/sam3_full_video/object_category_audit.json').read_text());excluded=set(o['sam_video_id'] for o in audit['excluded_objects']);n=len(raw);quality=np.zeros(n);orig_world=np.zeros((n,3));orig_local=np.zeros((n,3));Eall=np.zeros((n,4,4));Kall=np.zeros((n,3,3));timestamps=np.zeros(n,np.int64)
    for f in np.unique(frames):
        ix=np.flatnonzero(frames==f);meta={o['instance_id']:o for o in json.loads((base/'metadata'/f'{f:06d}.json').read_text())['observations']}
        with np.load(source/'frames'/f'{f:06d}.npz') as z:depth=z['camera_depth'];E=z['cam_extrinsic'].astype(float);K=z['cam_intrinsic'].astype(float);stamp=int(z['lidar_timestamp_ns'])
        inverse=np.linalg.inv(poses[f])
        for i in ix:
            o=meta[int(instances[i])];d=depth[np.array(o['selected_ids'],int)];spread=float(np.percentile(d,75)-np.percentile(d,25));quality[i]=min(len(d)/8,1)/(1+(spread/.5)**2);orig_world[i]=o['interior_xyz_map'];orig_local[i]=inverse[:3,:3]@orig_world[i]+inverse[:3,3];Eall[i]=E;Kall[i]=K;timestamps[i]=stamp
    references=np.full_like(raw,np.nan);available=np.ones(n,bool);sources=np.full(n,'pending',dtype='<U45');models=[];start=time.perf_counter()
    for tid in np.unique(ids):
        idx=np.flatnonzero(ids==tid)
        if tid in excluded:available[idx]=False;sources[idx]='excluded_visual_nonperson';continue
        # Long unseen gaps do not define motion through unobserved space.
        cuts=np.r_[0,np.flatnonzero(np.diff(timestamps[idx])*1e-9>2.)+1,len(idx)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            part=idx[lo:hi]
            if len(part)<3:available[part]=False;sources[part]='insufficient_continuous_track_evidence';continue
            t=(timestamps[part]-timestamps[part][0])*1e-9;fit,diag=robust_reference(t,raw[part],quality[part]);references[part]=fit;models.append(dict(sam_video_id=int(tid),first_frame=int(frames[part[0]]),last_frame=int(frames[part[-1]]),observations=len(part),**diag))
    static_models=[];stationary_ids=set()
    for tid in np.unique(ids):
        ix=np.flatnonzero((ids==tid)&available&np.isfinite(references).all(1))
        if len(ix)<10:continue
        origins=[];steps=[]
        for i in ix:
            E=Eall[i];inverse=np.linalg.inv(E);cam=E[:3,:3]@orig_local[i]+E[:3,3];T=poses[frames[i]]
            origins.append((T[:3,:3]@inverse[:3,3]+T[:3,3])[:2]);steps.append((T[:3,:3]@inverse[:3,:3]@(cam/cam[2]))[:2])
        t=(timestamps[ix]-timestamps[ix][0])*1e-9;centre,diag=static_reference(t,references[ix],quality[ix],origins,steps)
        if centre is not None:references[ix]=centre;stationary_ids.add(int(tid))
        static_models.append(dict(sam_video_id=int(tid),stationary_reference_used=centre is not None,**diag))
    fitting_seconds=time.perf_counter()-start;corrected=np.full_like(raw,np.nan);points_local=np.full((n,3),np.nan);selected_counts=np.zeros(n,int);depths=np.full(n,np.nan);reference_depths=np.full(n,np.nan);ray_origins=np.full((n,3),np.nan);ray_steps=np.full((n,3),np.nan);details=[];start=time.perf_counter()
    for f in np.unique(frames[available]):
        ix=np.flatnonzero(available&(frames==f));T=poses[f]
        with np.load(source/'frames'/f'{f:06d}.npz') as z:cloud=z['xyz'].astype(float);uv=z['pixels_uv'];visible=z['visible'];depth=z['camera_depth']
        with np.load(root/'map_review/sam3_full_video/masks'/f'{f:06d}.npz') as z:shape=tuple(z['mask_shape']);maskids=z['object_ids'];masks=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)
        pix=np.floor(np.where(np.isfinite(uv),uv,0)).astype(int);v=np.isfinite(cloud).all(1)&np.isfinite(uv).all(1)&visible&(depth>0)&(cloud>=[-20,-20,-3]).all(1)&(cloud<[20,20,6]).all(1)&(pix[:,0]>=0)&(pix[:,0]<shape[1])&(pix[:,1]>=0)&(pix[:,1]<shape[0]);valid=np.flatnonzero(v);cover=masks[:,pix[valid,1],pix[valid,0]].sum(0)
        for i in ix:
            mask=masks[np.flatnonzero(maskids==ids[i])[0]];support=valid[mask[pix[valid,1],pix[valid,0]]&(cover==1)]
            point,used,status,diag=depth_on_reference_tube(orig_local[i],references[i],T,Eall[i],cloud[support],support,tube=.5)
            if point is not None and int(ids[i]) in stationary_ids:
                # A static object's shared centre determines range; frame-specific
                # LiDAR depth variation must not reintroduce motion around it.
                diag['corrected_camera_depth_m']=diag['linear_camera_depth_m']
                point=np.asarray(diag['camera_ray_origin_lidar'])+diag['linear_camera_depth_m']*np.asarray(diag['camera_ray_step_lidar'])
                status='static_reference_camera_ray_projection'
            d=dict(frame=int(f),instance_id=int(instances[i]),sam_video_id=int(ids[i]),original_xy_map=raw[i].tolist(),reference_xy_map=references[i].tolist(),selected_support_ids=used.tolist(),source=status,**diag)
            if point is None:available[i]=False;sources[i]=status;details.append(d);continue
            points_local[i]=point;corrected[i]=(T[:3,:3]@point+T[:3,3])[:2];selected_counts[i]=len(used);sources[i]=status;depths[i]=diag['corrected_camera_depth_m'];reference_depths[i]=diag['linear_camera_depth_m'];ray_origins[i]=diag['camera_ray_origin_lidar'];ray_steps[i]=diag['camera_ray_step_lidar'];details.append(d)
    recovery_seconds=time.perf_counter()-start
    # Independent continuity check. Replace incompatible median-depth choices with the
    # closest point on the same camera ray; unresolved ray conflicts stay explicit ignores.
    def violations():
        result=[]
        for tid in np.unique(ids):
            ix=np.flatnonzero(available&(ids==tid))
            for i,j in zip(ix[:-1],ix[1:]):
                dt=(timestamps[j]-timestamps[i])*1e-9
                if dt<=2. and np.linalg.norm(corrected[j]-corrected[i])>3*dt+.35+1e-7:result.append((i,j))
        return result
    for _ in range(3):
        bad=violations()
        if not bad:break
        for i in set(k for pair in bad for k in pair):
            depths[i]=reference_depths[i];points_local[i]=ray_origins[i]+depths[i]*ray_steps[i];T=poses[frames[i]];corrected[i]=(T[:3,:3]@points_local[i]+T[:3,3])[:2];sources[i]='trajectory_regularized_camera_depth'
    ignored_continuity=[]
    for _ in range(n):
        bad=violations()
        if not bad:break
        involved=set(k for pair in bad for k in pair);i=max(involved,key=lambda k:(np.linalg.norm(corrected[k]-references[k]),-quality[k]));available[i]=False;sources[i]='ignored_camera_ray_continuity_conflict';corrected[i]=np.nan;ignored_continuity.append(i)
    assert not violations();residual=np.linalg.norm(corrected[available]-references[available],axis=1);assert residual.max(initial=0)<=.500001
    pixel_errors=[]
    for i in np.flatnonzero(available):
        old=Eall[i,:3,:3]@orig_local[i]+Eall[i,:3,3];new=Eall[i,:3,:3]@points_local[i]+Eall[i,:3,3];a=Kall[i]@old;b=Kall[i]@new;error=np.linalg.norm(a[:2]/a[2]-b[:2]/b[2]);assert error<1e-7 and new[2]>.1;pixel_errors.append(float(error))
    # Save final centres and their source; trajectory-only fills remain explicit.
    lookup={(d['frame'],d['instance_id']):d for d in details}
    for i in range(n):
        d=lookup.get((int(frames[i]),int(instances[i])),dict(frame=int(frames[i]),instance_id=int(instances[i]),sam_video_id=int(ids[i])))
        d.update(source=str(sources[i]),position_available=bool(available[i]),corrected_xy_map=corrected[i].tolist() if available[i] else None,current_camera_depth_m=float(depths[i]) if available[i] else None)
        (out/'metadata'/f'{frames[i]:06d}_I{instances[i]}.json').write_text(json.dumps(d,indent=2))
    np.savez_compressed(out/'centers_only.npz',original_xy_map=raw,reference_xy_map=references,corrected_xy_map=corrected,frame_ids=frames,instance_ids=instances,sam_video_ids=ids,timestamp_ns=timestamps,quality=quality,position_available=available,position_source=sources,selected_support_counts=selected_counts,original_point_lidar=orig_local,corrected_point_lidar=points_local,camera_depth=depths)
    counts={str(k):int((sources==k).sum()) for k in np.unique(sources)}
    report=dict(status='complete',input_positions=n,retained_positions=int(available.sum()),ignored_positions=int((~available).sum()),source_counts=counts,reference_models=models,static_models=static_models,parameters=dict(position_scale_m=.2,velocity_prior_scale_m_s=.5,acceleration_scale_m_s2=3.,max_reference_iterations=30,depth_iqr_scale_m=.5,reference_tube_m=.5,maximum_neighbour_jump='3m/s * elapsed_time +0.35m uncertainty',maximum_unseen_gap_s=2.,stationary_speed_threshold_m_s=.3,stationary_ray_inlier_fraction=.9,stationary_metric_fraction=.7,maximum_static_ray_condition=100),validation=dict(remaining_temporal_jump_outliers=0,max_reference_residual_m=float(residual.max(initial=0)),max_camera_pixel_change=max(pixel_errors,default=0),only_one_reference_per_continuous_segment=True),fitting_seconds=round(fitting_seconds,4),current_mask_depth_seconds=round(recovery_seconds,4),statistics=f'full {n} position population; no sampling',labels_modified=False,added_to_training=False,limits='No geometric outliers under the reported criteria; not an independent ground-truth accuracy guarantee. Camera-ray conflicts and insufficient-track evidence are ignored, not fabricated labels. Some retained positions are trajectory fills without sufficient current LiDAR support.')
    (out/'report.json').write_text(json.dumps(report,indent=2));render(root,out,raw,references,corrected,frames,ids,available,sources,timestamps);print(json.dumps({k:v for k,v in report.items() if k not in ['reference_models','static_models']},indent=2))


def render(root,out,raw,reference,corrected,frames,ids,available,sources,timestamps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualize_person_trajectories import draw_vehicles
    with np.load(root/'map_review/map_points.npz') as z:bg=z['background_xyz_map']
    vehicle=json.loads((root/'map_review/history_fit/report.json').read_text())['results']
    with np.load(root/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz') as z:keep=set(z['track_ids'].tolist())
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(18,14),dpi=140);ax.scatter(bg[:,1],bg[:,0],s=.15,c='#87939c',alpha=.3,linewidths=0);draw_vehicles(ax,[v for v in vehicle if v['track_id']in keep])
    for tid in np.unique(ids):
        s=(ids==tid)&available;rgb=colorsys.hsv_to_rgb(tid*.61803398875%1,.75,1.);ax.scatter(corrected[s,1],corrected[s,0],s=9,c=[rgb],alpha=.85,linewidths=0)
        if s.sum()>=3:p=np.median(corrected[s],axis=0);ax.annotate(f'P{tid}',(p[1],p[0]),xytext=(4,4),textcoords='offset points',color=rgb,fontsize=8)
    ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m), +X up');ax.grid(alpha=.15);ax.set_title(f'Consistent pedestrian track REVIEW · {available.sum()} retained centres\nOne reference per continuous segment · original camera pixel fixed · labels untouched');fig.tight_layout();fig.savefig(out/'whole_map_consistent.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,7),dpi=150)
    for ax,tid in zip(axes,[4,16]):
        ix=np.flatnonzero(ids==tid);good=ix[available[ix]];ax.scatter(raw[ix,1],raw[ix,0],c='#999',s=18,label='Original medians');ax.plot(reference[ix,1],reference[ix,0],c='#ffe16a',lw=1,label='Robust trajectory reference');ax.scatter(corrected[good,1],corrected[good,0],c='#00ffff',s=20,label='Ray-constrained final centres');ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m)');ax.set_ylabel('Map X (m)');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(f'P{tid}: {len(good)}/{len(ix)} centres retained')
    fig.tight_layout();fig.savefig(out/'P4_P16_before_after.png');plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{background:#15191d;color:white;font:18px sans-serif;margin:25px}a{color:cyan}img{max-width:100%}</style><h1>连续轨迹参考与相机射线约束</h1><p>移动目标每个连续观测片段使用一条稳健参考轨迹；同时满足低速度与多视角相机射线一致性的静止目标使用单一位置参考；偏离观测按轨迹约束重估深度。每帧中心像素固定。没有足够轨迹证据、非行人或无法满足几何约束的观测列为忽略，原始数据保留。部分位置为轨迹补点，并非人工真值。</p><p><a href="report.json">参数与全量检查</a> · <a href="centers_only.npz">位置、来源和可用标记</a></p><img src="whole_map_consistent.png"><img src="P4_P16_before_after.png">')


if __name__=='__main__':main()
