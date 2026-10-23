"""Accumulate raw-odom map and instance-supported vehicle points, without modifying labels."""
import argparse,colorsys,hashlib,json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def transform_points(points,T):return points@T[:3,:3].T+T[:3,3]


def color(tid):return (.65,.65,.65) if tid<0 else colorsys.hsv_to_rgb((tid*.61803398875)%1.,.75,.95)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--focus-track',type=int,default=8);args=parser.parse_args(argv)
    config=json.loads(args.config.read_text());labels=Path(config['temporal_output']);source=Path(config['source_index']).parent;rows=json.loads(Path(config['source_index']).read_text())['rows']
    output=labels/'map_review';(output/'instances').mkdir(parents=True,exist_ok=True)
    with np.load(config['pose_npz']) as z:poses=z['transforms']
    background=[];seen=set();vehicles=defaultdict(list);frame_ids=defaultdict(list);point_ids=defaultdict(list);accepted=defaultdict(list);bg_input=0;unassociated={};next_unassociated=-1
    for row,T in zip(rows,poses):
        frame=int(Path(row['path']).stem)
        with np.load(source/row['path']) as z:xyz=z['xyz'].astype(float);semantic=z['point_labels']
        valid=np.isfinite(xyz).all(1)&(np.abs(xyz[:,:2])<20).all(1)&(xyz[:,2]>=-3)&(xyz[:,2]<6)&(np.linalg.norm(xyz,axis=1)>=.5)
        context=valid&np.isin(semantic,[0,3,5,6,255])
        cloud=transform_points(xyz[context],T);bg_input+=len(cloud)
        keys=np.floor(cloud/.15).astype(np.int32)
        _,first=np.unique(keys,axis=0,return_index=True)
        keep=[]
        for i in first:
            key=tuple(keys[i])
            if key not in seen:seen.add(key);keep.append(i)
        if keep:background.append(cloud[keep].astype(np.float32))
        metadata=json.loads((labels/'metadata'/f'{frame:06d}.json').read_text())
        for item in metadata['observations']:
            if item['kind']!='vehicle':continue
            tid=item.get('track_id')
            if tid is None:
                tid=next_unassociated;next_unassociated-=1;unassociated[tid]=(frame,item['seed_id'])
            tid=int(tid);ids=np.asarray(item['support_ids'],np.int32)
            if not len(ids):continue
            if not np.isfinite(xyz[ids]).all():raise ValueError(f'Nonfinite support points at {frame}')
            world=transform_points(xyz[ids],T).astype(np.float32)
            vehicles[tid].append(world);frame_ids[tid].append(np.full(len(ids),frame,np.int32));point_ids[tid].append(ids)
            accepted[tid].append(np.full(len(ids),item['status'] in ['accepted_temporal','accepted_single_frame'],bool))
        if (frame+1)%100==0:print(f'mapped {frame+1}/{len(rows)}',flush=True)
    bg=np.concatenate(background) if background else np.empty((0,3),np.float32)
    summaries=[];all_xyz=[];all_frames=[];all_ids=[];all_tracks=[];all_accepted=[]
    for tid,chunks in sorted(vehicles.items()):
        xyz=np.concatenate(chunks);frames=np.concatenate(frame_ids[tid]);ids=np.concatenate(point_ids[tid]);saved=np.concatenate(accepted[tid]);rgb=color(tid)
        label=f'T{tid}' if tid>=0 else f'U{unassociated[tid][0]:06d}/{unassociated[tid][1]}'
        filename=f'T{tid:03d}.npz' if tid>=0 else f'U{unassociated[tid][0]:06d}_{unassociated[tid][1]}.npz'
        np.savez_compressed(output/'instances'/filename,xyz_map=xyz,frame_ids=frames,source_point_indices=ids,saved_label_observation=saved)
        summaries.append(dict(track_id=tid,label=label,identity_status='tracked_vehicle' if tid>=0 else 'unassociated_diagnostic_observation',points=len(xyz),frames=len(np.unique(frames)),first_frame=int(frames.min()),last_frame=int(frames.max()),xyz_bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],display_centroid=np.median(xyz,axis=0).tolist(),color_rgb=[round(v*255) for v in rgb],saved_label_support_points=int(saved.sum()),file=f'instances/{filename}'))
        all_xyz.append(xyz);all_frames.append(frames);all_ids.append(ids);all_tracks.append(np.full(len(xyz),tid,np.int32));all_accepted.append(saved)
    car=np.concatenate(all_xyz) if all_xyz else np.empty((0,3),np.float32);frames=np.concatenate(all_frames) if all_frames else np.empty((0,),np.int32);tracks=np.concatenate(all_tracks) if all_tracks else np.empty((0,),np.int32);ids=np.concatenate(all_ids) if all_ids else np.empty((0,),np.int32);saved=np.concatenate(all_accepted) if all_accepted else np.empty((0,),bool)
    np.savez_compressed(output/'map_points.npz',background_xyz_map=bg,vehicle_xyz_map=car,vehicle_track_ids=tracks,vehicle_frame_ids=frames,vehicle_source_point_indices=ids,vehicle_saved_label_observation=saved,odom_trajectory=poses[:,:3,3].astype(np.float32))
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(17,13),dpi=140)
    ax.scatter(bg[:,1],bg[:,0],s=.15,c='#84919a',alpha=.3,linewidths=0,rasterized=True)
    ax.plot(poses[:,1,3],poses[:,0,3],c='white',lw=.8,alpha=.75,label='Recorded odom path')
    occupied=[];offset=max(1.5,float(np.ptp(bg[:,:2],axis=0).max())*.025) if len(bg) else 1.5
    for record in summaries:
        tid=record['track_id'];points=np.concatenate(vehicles[tid]);rgb=color(tid);ax.scatter(points[:,1],points[:,0],s=1.6,c=[rgb],alpha=.7,linewidths=0,rasterized=True)
        x,y=record['display_centroid'][:2];tx,ty=y,x
        for dx,dy in [(0,offset),(offset,0),(-offset,0),(0,-offset),(offset,offset),(-offset,-offset),(2*offset,offset)]:
            px,py=tx+dx,ty+dy
            if all(abs(px-a)>offset or abs(py-b)>offset*.7 for a,b in occupied):break
        occupied.append((px,py));ax.annotate(record['label'],xy=(tx,ty),xytext=(px,py),color=rgb,fontsize=10,arrowprops=dict(arrowstyle='-',color=rgb,lw=.6),bbox=dict(facecolor='#15191d',alpha=.75,edgecolor='none',pad=2))
    ax.scatter([poses[0,1,3]],[poses[0,0,3]],marker='^',s=55,c='white',label='Start')
    ax.set_aspect('equal');ax.invert_xaxis();ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m), +X up');ax.grid(alpha=.13)
    ax.set_title(f'ZGC · {len(rows)} frames · recorded-odom accumulation\nVehicle instance support points colored by track ID; not an optimized map',fontsize=15)
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(output/'whole_map_bev.png');plt.close(fig)
    # Preserve all car points; background-only thinning is for the interactive display.
    rng=np.random.default_rng(20261014);choice=rng.choice(len(bg),min(100000,len(bg)),replace=False)
    p=bg[choice];figure=go.Figure([go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name='Background',marker=dict(size=1,color='#87939c',opacity=.2),hoverinfo='skip')])
    for record in summaries:
        tid=record['track_id'];xyz=np.concatenate(vehicles[tid]);f=np.concatenate(frame_ids[tid]);rgb=record['color_rgb']
        figure.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',name=f'{record["label"]} · {record["frames"]} frames',customdata=f,hovertemplate=f'{record["label"]}<br>frame %{{customdata}}<br>X %{{x:.2f}} Y %{{y:.2f}} Z %{{z:.2f}}<extra></extra>',marker=dict(size=2,color=f'rgb({rgb[0]},{rgb[1]},{rgb[2]})',opacity=.65)))
    figure.add_trace(go.Scatter3d(x=poses[:,0,3],y=poses[:,1,3],z=poses[:,2,3],mode='lines',name='Odom path',line=dict(color='white',width=3)))
    figure.update_layout(template='plotly_dark',title='Recorded-odom map · click legend to isolate a vehicle track',scene=dict(aspectmode='data',xaxis_title='Map X (m)',yaxis_title='Map Y (m)',zaxis_title='Map Z (m)'),margin=dict(l=0,r=0,b=0,t=50))
    figure.write_html(output/'whole_map_3d.html',include_plotlyjs=True)
    # White SUV of frames 676/677: color by acquisition time to reveal temporal disagreement.
    focus=args.focus_track if args.focus_track in vehicles else (min(vehicles) if vehicles else None)
    if focus is not None:
        xyz=np.concatenate(vehicles[focus]);f=np.concatenate(frame_ids[focus]);centre=np.median(xyz,axis=0)
        fig,axes=plt.subplots(1,3,figsize=(18,6),dpi=140)
        for ax,(u,v,labels2) in zip(axes,[(1,0,('Map Y (m)','Map X (m)')),(0,2,('Map X (m)','Map Z (m)')),(1,2,('Map Y (m)','Map Z (m)'))]):
            im=ax.scatter(xyz[:,u],xyz[:,v],c=f,s=2,cmap='turbo',alpha=.55,linewidths=0,rasterized=True);ax.set_aspect('equal');ax.set_xlabel(labels2[0]);ax.set_ylabel(labels2[1]);ax.grid(alpha=.15)
        axes[0].invert_xaxis();fig.colorbar(im,ax=axes.tolist(),label='Source frame ID',shrink=.7,pad=.025);fig.suptitle(f'T{focus}: all instance-supported points · {len(np.unique(f))} frames · color by time\nDirect recorded-odom transform; no point-cloud registration or label changes')
        fig.savefig(output/f'T{focus:03d}_temporal_views.png',bbox_inches='tight');plt.close(fig)
        # Local context around the focus car uses every map point inside a 10 m XY neighbourhood.
        nearby=(np.abs(bg[:,:2]-centre[:2])<=10).all(1);fig,ax=plt.subplots(figsize=(11,9),dpi=140)
        ax.scatter(bg[nearby,1],bg[nearby,0],s=.6,c='#87939c',alpha=.25,linewidths=0)
        im=ax.scatter(xyz[:,1],xyz[:,0],c=f,s=3,cmap='turbo',alpha=.6,linewidths=0);fig.colorbar(im,ax=ax,label='Source frame ID')
        ax.set_xlim(centre[1]+10,centre[1]-10);ax.set_ylim(centre[0]-10,centre[0]+10);ax.set_aspect('equal');ax.set_xlabel('Map Y (m), +Y left');ax.set_ylabel('Map X (m), +X up');ax.set_title(f'T{focus} white SUV · accumulated cloud with map context');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(output/f'T{focus:03d}_map_context.png');plt.close(fig)
    manifest=dict(status='complete',frames=len(rows),vehicle_tracks=sum(r['track_id']>=0 for r in summaries),unassociated_observations=len(unassociated),vehicle_support_points=len(car),background_points_before_voxel=bg_input,background_voxel_points=len(bg),background_voxel_m=.15,interactive_background_sampling='uniform random without replacement, seed 20261014, at most100000 voxel points',interactive_background_points=len(choice),vehicle_point_sampling='none; all supported points retained and drawn',local_source_roi='each scan XY(-20,20)m, Z[-3,6)m; map extent covers entire trajectory',background_semantic_ids=[0,3,5,6,255],focus_track=focus,instances=summaries,pose_sha256=hashlib.sha256(Path(config['pose_npz']).read_bytes()).hexdigest(),coordinate_assumption=config['coordinate_assumption'],labels_modified=False,manual_gt=False,limits='Track IDs are current association IDs, not guaranteed unique physical vehicles. Map uses recorded odom and its drift; no SLAM/loop correction has been applied.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='instances'},indent=2))


if __name__=='__main__':main()
