"""Per-frame camera/LiDAR review of consistent persons and deduplicated static cars."""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw
from render_combined import EDGES,clip_bev_edge,boxes_to_corners_lidar
from projection_clipping import project_clipped_box_edges
from render_temporal import font


def initialize(config_path,output=None):
    global C,ROOT,SOURCE,OUT,POSES,PERSONS,CARS,read_source_image
    C=json.loads(Path(config_path).read_text());ROOT=Path(C['temporal_output']);SOURCE=Path(C['source_index']).parent
    OUT=Path(output) if output else ROOT/'combined_consistent_review';sys.path.insert(0,str(Path(__file__).resolve().parent))
    from common import read_source_image
    with np.load(C['pose_npz']) as z:POSES=z['transforms']
    person=ROOT/'map_review/sam3_full_video/interior_half_review/consistent_tracks'
    from finalize_person_positions import load_positions
    PERSONS=load_positions(person/'centers_only.npz')
    with np.load(ROOT/'map_review/history_fit/bev_dedup/vehicle_map_boxes.npz') as z:
        CARS=list(zip(z['track_ids'].copy(),z['box_3d_map'].reshape(-1,7).copy()))


def render(task):
    frame,row=task;stem=f'{frame:06d}';T=POSES[frame];inverse=np.linalg.inv(T)
    image=read_source_image(row).convert('RGB');width,height=image.size
    with np.load(SOURCE/row['path']) as z:
        arrays={key:z[key] for key in ('xyz','pixels_uv','visible','camera_depth','cam_extrinsic','cam_intrinsic','point_raw_labels','point_labels','training_roi_mask') if key in z}
        xyz=arrays['xyz'];uv=arrays['pixels_uv'];visible=arrays['visible'];E=arrays['cam_extrinsic'];K=arrays['cam_intrinsic']
    ix=np.flatnonzero((PERSONS['frame_ids']==frame)&PERSONS['position_available'])
    video=ROOT/'map_review/sam3_full_video'
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from tools.lidar_annotator.track_preview import load_targets
    from tools.lidar_annotator.display_range import DEFAULT_LIMITS
    job=ROOT.parent;visual=job/'data/visual_instances';records_path=visual/f'{stem}.json'
    records=json.loads(records_path.read_text()).get('records',[]) if records_path.is_file() else []
    mapping_path=video/'native_to_display_ids.json';mapping=json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    for record in records:record['display_id']=mapping.get(str(record.get('track_id')),record.get('track_id'))
    paths=dict(local=frame,tracks=video/'interior_half_review/consistent_tracks/centers_only.npz',
               person_metadata=Path(C['person_root'])/'metadata'/f'{stem}.json',id_map=mapping_path,
               instance_masks=visual/f'{stem}.npz',track_masks=video/'masks'/f'{stem}.npz',mask=SOURCE/'image_masks'/f'{stem}.png')
    support_targets={target['id']:target for target in load_targets(paths,arrays,records,DEFAULT_LIMITS) if target['prompt']=='person'}
    with np.load(video/'masks'/f'{stem}.npz') as z:
        shape=tuple(z['mask_shape']);maskids=z['object_ids'];masks=np.unpackbits(z['masks_packed'],axis=1,count=int(np.prod(shape)),bitorder='little').reshape(-1,*shape).astype(bool)
    overlay=np.zeros((height,width,4),np.uint8);pitems=[]
    for i in ix:
        tid=int(PERSONS['sam_video_ids'][i]);mask=masks[np.flatnonzero(maskids==tid)[0]];yy,xx=np.nonzero(mask)
        if not len(xx):continue
        overlay[mask]=(0,255,255,38)
        target=support_targets.get(tid)
        pitems.append((i,tid,(xx.min(),yy.min(),xx.max(),yy.max()),target['optimized_indices'] if target else np.empty(0,int)))
    image=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(overlay)).convert('RGB');image.thumbnail((1200,675))
    canvas=Image.new('RGB',(2080,1020),(20,23,28));canvas.paste(image,(20,90));draw=ImageDraw.Draw(canvas)
    def pixel(p):return (20+p[0]*image.width/width,90+p[1]*image.height/height)
    def bev(p):return (1260+(20-p[1])*20,70+(20-p[0])*20)
    for v in range(-20,21,5):
        x=1260+(20-v)*20;y=70+(20-v)*20
        draw.line((x,70,x,870),fill=(50,55,62));draw.line((1260,y,2060,y),fill=(50,55,62))
        draw.text((x-12,878),str(v),font=font(17),fill='white');draw.text((1222,y-10),str(v),font=font(17),fill='white')
    valid=np.isfinite(xyz).all(1)&(np.abs(xyz[:,:2])<20).all(1)
    for p in xyz[valid]:
        x,y=bev(p);draw.point((x,y),fill=(105,110,115))
    draw.rectangle((1260,70,2060,870),outline=(170,175,180));draw.text((1270,80),'X (m) +X up',font=font(18),fill='white');draw.text((1540,912),'Y (m) +Y left',font=font(19),fill='white')
    def support(ids,color):
        ids=ids[(ids>=0)&(ids<len(xyz))]
        for pid in ids:
            if valid[pid]:
                x,y=bev(xyz[pid]);draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            if visible[pid] and np.isfinite(uv[pid]).all() and 0<=uv[pid,0]<width and 0<=uv[pid,1]<height:
                x,y=pixel(uv[pid]);draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    cars=[];label_bounds=[]
    def label(anchor,text,color):
        x=max(22,min(1100,anchor[0]));y=max(92,min(740,anchor[1]));f=font(17)
        for _ in range(12):
            bounds=draw.textbbox((x,y),text,font=f)
            if not any(bounds[0]<b[2] and bounds[2]>b[0] and bounds[1]<b[3] and bounds[3]>b[1] for b in label_bounds):break
            y+=22
        label_bounds.append(draw.textbbox((x,y),text,font=f));draw.text((x,y),text,font=f,fill=color,stroke_width=2,stroke_fill='black')
    for tid,world in CARS:
        box=world.copy();box[:3]=inverse[:3,:3]@world[:3]+inverse[:3,3];box[6]=world[6]-np.arctan2(T[1,0],T[0,0])
        corners=boxes_to_corners_lidar(box[None])[0]
        if corners[:,0].min()>20 or corners[:,0].max()<-20 or corners[:,1].min()>20 or corners[:,1].max()<-20:continue
        # Spatially associate raw current scan returns for display only, not fitting.
        delta=xyz-box[:3];co,si=np.cos(box[6]),np.sin(box[6]);local=np.c_[co*delta[:,0]+si*delta[:,1],-si*delta[:,0]+co*delta[:,1],delta[:,2]]
        pointids=np.flatnonzero(np.isfinite(local).all(1)&(np.abs(local)<=box[3:6]/2+.12).all(1))
        color=(255,155,30);cam=corners@E[:3,:3].T+E[:3,3]
        # Do not draw an occluded/unobserved map car through a building in RGB.
        image_supported=int(visible[pointids].sum())>=3
        edges=project_clipped_box_edges(cam,EDGES,K,width,height) if image_supported else []
        for _,a,b in edges:draw.line((*pixel(a),*pixel(b)),fill=color,width=2)
        polygon=[bev(corners[j]) for j in (0,1,5,4)]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            edge=clip_bev_edge(a,b)
            if edge:draw.line((*edge[0],*edge[1]),fill=color,width=2)
        x,y=bev(box)
        if -20<=box[0]<=20 and -20<=box[1]<=20:draw.text((min(1970,x+5),max(75,min(845,y+5))),f'V{tid}',font=font(17),fill=color)
        if edges:label(pixel(edges[0][1]),f'V{tid} map',color)
        support(pointids,color);cars.append(dict(track_id=int(tid),current_scan_points=len(pointids),box_lidar=box.tolist(),source='history_fit_deduplicated_static_map'))
    for i,tid,bbox,pointids in pitems:
        color=(0,255,255);a=pixel(bbox[:2]);b=pixel(bbox[2:]);draw.rectangle((*a,*b),outline=color,width=2)
        src=str(PERSONS['position_source'][i]);suffix=' track-only' if PERSONS['track_only'][i] else ''
        label((a[0],a[1]-20),f'P{tid}{suffix} n={len(pointids)}',color)
        support(pointids,color);point=PERSONS['corrected_point_lidar'][i];cam=E[:3,:3]@point+E[:3,3];h=K@cam
        if cam[2]>.1:
            p=h[:2]/h[2]
            if 0<=p[0]<width and 0<=p[1]<height:
                x,y=pixel(p);draw.line((x-7,y,x+7,y),fill='white',width=2);draw.line((x,y-7,x,y+7),fill='white',width=2)
        if (np.abs(point[:2])<20).all():
            x,y=bev(point);draw.line((x-7,y,x+7,y),fill=color,width=3);draw.line((x,y-7,x,y+7),fill=color,width=3);draw.text((min(1970,x+7),max(75,min(845,y+7))),f'P{tid}{suffix}',font=font(17),fill=color)
    draw.text((22,25),f'{stem}  Optimized REVIEW | camera masks + LiDAR',font=font(25),fill='white')
    draw.text((1260,30),f'BEV +/-20m | {len(cars)} map cars / {len(pitems)} persons',font=font(23),fill='white')
    for y,text in [(805,'Cyan: SAM3 person mask / selected LiDAR / optimized centre'),(838,'Orange: history-fit static vehicles projected from deduplicated map'),(871,'P ID: SAM3 video ID | V ID: vehicle map track | fill: trajectory-only')]:draw.text((22,y),text,font=font(21),fill='white')
    draw.text((1260,950),'Gray: full current scan | colour: current scan supports',font=font(18),fill='white')
    draw.text((1260,982),'Review candidates | original labels unchanged',font=font(18),fill='white')
    dest=OUT/'visualization'/f'{stem}.png';canvas.save(dest)
    with Image.open(dest) as check:check.verify()
    record=dict(frame=frame,persons=[dict(sam_video_id=tid,source=str(PERSONS['position_source'][i]),support_points=len(s),
                track_only=bool(PERSONS['track_only'][i]),position_policy=PERSONS['position_policy'],position_lidar=PERSONS['corrected_point_lidar'][i].tolist(),
                support_status=(support_targets.get(tid,{}).get('semantic_selection') or {}).get('status','missing_support'),
                support_review_required=(support_targets.get(tid,{}).get('semantic_selection') or {}).get('review_required',True)) for i,tid,_,s in pitems],vehicles=cars)
    (OUT/'visualization'/f'{stem}.json').write_text(json.dumps(record,indent=2));return frame,len(pitems),len(cars)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--frames',nargs='+',type=int);p.add_argument('--output',type=Path);a=p.parse_args();initialize(a.config,a.output)
    (OUT/'visualization').mkdir(parents=True,exist_ok=True);rows=json.loads(Path(C['source_index']).read_text())['rows'];tasks=[(int(Path(r['path']).stem),r) for r in rows if a.frames is None or int(Path(r['path']).stem) in a.frames]
    known={int(Path(r['path']).stem) for r in rows}
    if a.frames is not None and set(a.frames)-known:p.error(f'Unknown frames: {sorted(set(a.frames)-known)}')
    if a.workers<1:p.error('--workers must be positive')
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.config,str(OUT))) as pool:
        futures=[pool.submit(render,t) for t in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%50==0:print(f'rendered {len(records)}/{len(tasks)}',flush=True)
    records.sort();html='<!doctype html><meta charset="utf-8"><title>ZGC combined review</title><style>body{background:#14171c;color:white;font:18px sans-serif}img{width:100%;max-width:1560px}a{color:cyan}</style><h1>ZGC 逐帧行人、车辆与激光点云</h1><p>最新行人位置；去重后的静止车辆地图回投；灰色为当前帧全点云。所有结果供审核。</p>'
    html+=''.join(f'<section id="f{f:06d}"><h2>{f:06d} | {nperson} persons / {ncar} map cars</h2><a href="visualization/{f:06d}.png"><img loading="lazy" src="visualization/{f:06d}.png"></a></section>' for f,nperson,ncar in records)
    (OUT/'index.html').write_text(html)
    report=dict(status='complete',frames=len(records),frame_ids=[r[0] for r in records],person_positions=sum(r[1] for r in records),vehicle_map_projections=sum(r[2] for r in records),elapsed_seconds=round(time.perf_counter()-start,2),labels_modified=False,vehicle_source='history-fit BEV-deduplicated static map; current box-contained LiDAR points are display supports, not verified instance segmentation',pedestrian_source='consistent_tracks centres and selected current-frame mask supports',bev_range_m=[-20,20])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
